use qwen3::template::{TemplateError, render_chat, template_hash};

#[test]
fn chat_template_render() {
    let messages = [
        ("system", "You are Alloy's test assistant."),
        ("user", "Say exactly: ok"),
    ];
    assert_eq!(
        render_chat(&messages).expect("render"),
        include_str!("../fixture/golden_render.txt")
    );
}

#[test]
fn template_hash_identity() {
    let expected = include_str!("../fixture/chat_template.sha256").trim();
    assert_eq!(template_hash(), expected);
    render_chat(&[("user", "different content")]).expect("render");
    assert_eq!(template_hash(), expected);
}

#[test]
fn invalid_messages_return_errors() {
    assert_eq!(render_chat(&[]), Err(TemplateError::InvalidMessages));
    assert_eq!(
        render_chat(&[("tool", "unsupported")]),
        Err(TemplateError::InvalidMessages)
    );
}

#[test]
fn content_is_not_interpreted_as_template_source() {
    let output = render_chat(&[("user", "{{ 7 * 7 }} <&>")]).expect("render");
    assert!(output.contains("{{ 7 * 7 }} <&>"));
}
