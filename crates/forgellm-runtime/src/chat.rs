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
}
