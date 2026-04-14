//! Chat template formatting for instruct models.
//!
//! Formats conversations into the prompt format expected by each model family.
//! Templates are based on the model's architecture and common conventions.

/// A message in a conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

/// Chat template format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChatTemplate {
    /// SmolLM/Llama-style: `<|im_start|>role\ncontent<|im_end|>`
    ChatML,
    /// Llama 3 style: `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
    Llama3,
    /// Qwen style (same as ChatML)
    Qwen,
    /// Raw: just concatenate messages with no special formatting
    Raw,
}

impl ChatTemplate {
    /// Detect the appropriate template from a model architecture name.
    pub fn from_architecture(arch: &str) -> Self {
        match arch.to_lowercase().as_str() {
            "llama" => ChatTemplate::ChatML, // SmolLM and many Llama finetunes use ChatML
            "qwen2" => ChatTemplate::Qwen,
            "mistral" => ChatTemplate::Raw, // Mistral uses [INST] but we simplify
            _ => ChatTemplate::ChatML,
        }
    }

    /// Format a list of messages into a prompt string.
    pub fn format(&self, messages: &[ChatMessage]) -> String {
        match self {
            ChatTemplate::ChatML | ChatTemplate::Qwen => format_chatml(messages),
            ChatTemplate::Llama3 => format_llama3(messages),
            ChatTemplate::Raw => format_raw(messages),
        }
    }

    /// Format a single user prompt as a chat conversation.
    pub fn format_prompt(&self, prompt: &str) -> String {
        self.format(&[ChatMessage::user(prompt)])
    }

    /// Format a single user prompt with a system message.
    pub fn format_with_system(&self, system: &str, prompt: &str) -> String {
        self.format(&[ChatMessage::system(system), ChatMessage::user(prompt)])
    }
}

/// ChatML format: `<|im_start|>role\ncontent<|im_end|>\n`
fn format_chatml(messages: &[ChatMessage]) -> String {
    let mut output = String::new();
    for msg in messages {
        output.push_str("<|im_start|>");
        output.push_str(&msg.role);
        output.push('\n');
        output.push_str(&msg.content);
        output.push_str("<|im_end|>\n");
    }
    output.push_str("<|im_start|>assistant\n");
    output
}

/// Llama 3 format
fn format_llama3(messages: &[ChatMessage]) -> String {
    let mut output = String::from("<|begin_of_text|>");
    for msg in messages {
        output.push_str("<|start_header_id|>");
        output.push_str(&msg.role);
        output.push_str("<|end_header_id|>\n\n");
        output.push_str(&msg.content);
        output.push_str("<|eot_id|>");
    }
    output.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    output
}

/// Raw format: just concatenate
fn format_raw(messages: &[ChatMessage]) -> String {
    let mut output = String::new();
    for msg in messages {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(&msg.content);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_single_user() {
        let template = ChatTemplate::ChatML;
        let result = template.format_prompt("Hello");
        assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn chatml_with_system() {
        let template = ChatTemplate::ChatML;
        let result = template.format_with_system("You are helpful.", "Hello");
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
    }

    #[test]
    fn chatml_multi_turn() {
        let template = ChatTemplate::ChatML;
        let messages = vec![
            ChatMessage::user("What is Rust?"),
            ChatMessage::assistant("A systems programming language."),
            ChatMessage::user("Tell me more."),
        ];
        let result = template.format(&messages);
        assert!(result.contains("What is Rust?"));
        assert!(result.contains("A systems programming language."));
        assert!(result.contains("Tell me more."));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn llama3_format() {
        let template = ChatTemplate::Llama3;
        let result = template.format_prompt("Hello");
        assert!(result.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(result.contains("Hello"));
        assert!(result.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn detect_from_architecture() {
        assert_eq!(
            ChatTemplate::from_architecture("llama"),
            ChatTemplate::ChatML
        );
        assert_eq!(ChatTemplate::from_architecture("qwen2"), ChatTemplate::Qwen);
    }

    #[test]
    fn raw_format() {
        let template = ChatTemplate::Raw;
        let messages = vec![ChatMessage::user("Hello"), ChatMessage::user("World")];
        let result = template.format(&messages);
        assert_eq!(result, "Hello\nWorld");
    }

    // ── Real-world validation tests ──────────────────────────────────────

    #[test]
    fn chatml_empty_messages_produces_assistant_header() {
        // An empty message list should still produce the assistant prompt header
        // so the model knows to start generating.
        let template = ChatTemplate::ChatML;
        let result = template.format(&[]);
        assert_eq!(
            result, "<|im_start|>assistant\n",
            "empty messages should produce just the assistant header"
        );
    }

    #[test]
    fn llama3_empty_messages_produces_assistant_header() {
        let template = ChatTemplate::Llama3;
        let result = template.format(&[]);
        assert!(
            result.contains("<|start_header_id|>assistant<|end_header_id|>"),
            "empty Llama3 messages should still produce assistant header"
        );
    }

    #[test]
    fn chatml_handles_special_characters_in_content() {
        // Content with newlines, angle brackets, and pipe characters should
        // be preserved verbatim (no escaping in ChatML).
        let template = ChatTemplate::ChatML;
        let content = "Here is code:\n```rust\nfn main() { println!(\"<|test|>\"); }\n```";
        let result = template.format_prompt(content);
        assert!(
            result.contains(content),
            "special characters in content should be preserved verbatim"
        );
    }

    #[test]
    fn chatml_multi_turn_preserves_order() {
        // A multi-turn conversation should maintain message order and all
        // role transitions should be correct.
        let template = ChatTemplate::ChatML;
        let messages = vec![
            ChatMessage::system("You are a calculator."),
            ChatMessage::user("What is 2+2?"),
            ChatMessage::assistant("4"),
            ChatMessage::user("And 3+3?"),
        ];
        let result = template.format(&messages);

        // Verify ordering: system before first user, assistant before second user
        let sys_pos = result.find("system\nYou are a calculator.").unwrap();
        let user1_pos = result.find("user\nWhat is 2+2?").unwrap();
        let asst_pos = result.find("assistant\n4").unwrap();
        let user2_pos = result.find("user\nAnd 3+3?").unwrap();
        let final_asst = result.rfind("<|im_start|>assistant\n").unwrap();

        assert!(sys_pos < user1_pos, "system should come before first user");
        assert!(
            user1_pos < asst_pos,
            "first user should come before assistant response"
        );
        assert!(
            asst_pos < user2_pos,
            "assistant response should come before second user"
        );
        assert!(
            user2_pos < final_asst,
            "second user should come before final assistant prompt"
        );
    }

    #[test]
    fn from_architecture_unknown_defaults_to_chatml() {
        // Unknown architectures should default to ChatML rather than panicking.
        let template = ChatTemplate::from_architecture("unknown_arch_xyz");
        assert_eq!(template, ChatTemplate::ChatML);
    }
}
