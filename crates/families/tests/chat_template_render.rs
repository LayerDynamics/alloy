use qwen3::template::render_chat;

/// Byte-identical golden render; replaced when the template lands.
const GOLDEN: &str = include_str!("../fixture/golden_render.txt");

#[test]
fn chat_template_render() {
    let messages = [
        ("system", "You are Alloy's test assistant."),
        ("user", "Say exactly: ok"),
    ];
    let rendered = render_chat(&messages).expect("template renders");
    assert_eq!(
        rendered, GOLDEN,
        "rendered chat must be byte-identical to the committed golden render"
    );
}