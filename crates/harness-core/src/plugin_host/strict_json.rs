//! Reject ambiguous JSON before any typed response deserialization.
use super::HostError;
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};
use std::fmt;

struct Seed(usize);
impl<'de> DeserializeSeed<'de> for Seed {
    type Value = Value;
    fn deserialize<D: de::Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        if self.0 > 64 {
            return Err(de::Error::custom("hook JSON nesting limit"));
        }
        deserializer.deserialize_any(self)
    }
}
impl<'de> Visitor<'de> for Seed {
    type Value = Value;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("unambiguous JSON")
    }
    fn visit_bool<E: de::Error>(self, v: bool) -> Result<Value, E> {
        Ok(Value::Bool(v))
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Value, E> {
        Ok(v.into())
    }
    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Value, E> {
        Ok(v.into())
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Value, E> {
        serde_json::Number::from_f64(v)
            .map(Value::Number)
            .ok_or_else(|| E::custom("nonfinite JSON"))
    }
    fn visit_str<E: de::Error>(self, v: &str) -> Result<Value, E> {
        Ok(Value::String(v.into()))
    }
    fn visit_string<E: de::Error>(self, v: String) -> Result<Value, E> {
        Ok(Value::String(v))
    }
    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(v) = seq.next_element_seed(Seed(self.0 + 1))? {
            values.push(v);
        }
        Ok(Value::Array(values))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom("duplicate hook JSON key"));
            }
            values.insert(key, map.next_value_seed(Seed(self.0 + 1))?);
        }
        Ok(Value::Object(values))
    }
}
pub(super) fn decode<T: de::DeserializeOwned>(bytes: &[u8]) -> Result<T, HostError> {
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    let value = Seed(0)
        .deserialize(&mut decoder)
        .map_err(|_| HostError::new("ambiguous, malformed or over-depth hook JSON"))?;
    decoder
        .end()
        .map_err(|_| HostError::new("trailing hook JSON"))?;
    serde_json::from_value(value).map_err(|_| HostError::new("invalid hook response contract"))
}
#[cfg(test)]
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Empty {}
#[test]
fn duplicate_escaped_nested_and_trailing_json_rejects() {
    for input in [
        r#"{"x":1,"x":2}"#,
        r#"{"x":1,"\u0078":2}"#,
        r#"{"annotations":{"a.x":1,"a.x":2}}"#,
        r#"{"nested":[{"x":1,"x":2}]}"#,
        "{} {}",
    ] {
        assert!(decode::<Value>(input.as_bytes()).is_err(), "{input}");
    }
    assert!(decode::<Empty>(b"{}").is_ok());
    assert!(decode::<Empty>(b"{\"unknown\":0}").is_err());
    assert!(decode::<Value>(format!("{}0{}", "[".repeat(70), "]".repeat(70)).as_bytes()).is_err());
}
