use async_trait::async_trait;
use dashmap::DashMap;
use derive_builder::Builder;
use futures::stream::BoxStream;
use regex::Regex;

pub struct StopConditionFactory {
    stop_regex_cache: DashMap<String, Regex>,
}

fn reverse<T>(s: T) -> String
    where
        T: Into<String>,
{
    s.into().chars().rev().collect()
}

impl Default for StopConditionFactory {
    fn default() -> Self {
        Self {
            stop_regex_cache: DashMap::new(),
        }
    }
}

fn create_stop_regex(stop_words: Vec<String>) -> Regex {
    // (?m) enables multi-line matching mode.
    // \A means absolute begins of string.
    let reversed_stop_words: Vec<_> = stop_words
        .iter()
        .map(|x| regex::escape(&reverse(x)))
        .collect();
    let regex_string = r"(?m)\A".to_owned() + "((" + &reversed_stop_words.join(")|(") + "))";
    Regex::new(&regex_string).expect("Failed to create regex")
}

pub struct StopCondition {
    stop_re: Option<Regex>,
    max_decoding_length: usize,
    reversed_text: String,
    num_decoded: usize,
}

impl StopCondition {
    pub fn new(stop_re: Option<Regex>, max_decoding_length: usize, text: &str) -> Self {
        Self {
            stop_re,
            max_decoding_length,
            reversed_text: reverse(text),
            num_decoded: 0,
        }
    }

    pub fn should_stop(&mut self, new_text: &str) -> bool {
        if !new_text.is_empty() {
            self.reversed_text = reverse(new_text) + &self.reversed_text;

            if let Some(re) = &self.stop_re {
                if re.is_match(&self.reversed_text) {
                    return true;
                }
            }
        }

        self.num_decoded += 1;
        self.num_decoded >= self.max_decoding_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        let text = reverse("void write_u32(std::uint32_t val) const {\n        write_raw(&val, sizeof(val));\n    }\n\n    ~llama_file() {\n        if (fp) {\n            std::fclose(fp);\n        }\n    }\n};\n\nvoid");
        assert!(!create_stop_regex(vec!["\n\n".to_owned(), "\n\n  ".to_owned()]).is_match(&text));
        assert!(create_stop_regex(vec![
            "\n\n".to_owned(),
            "\n\n  ".to_owned(),
            "\nvoid".to_owned()
        ])
            .is_match(&text));
    }
}

#[derive(Builder, Debug)]
pub struct TextGenerationOptions {
    #[builder(default = "1024")]
    pub max_input_length: usize,

    #[builder(default = "256")]
    pub max_decoding_length: usize,

    #[builder(default = "1.0")]
    pub sampling_temperature: f32,
}

#[async_trait]
pub trait TextGeneration: Sync + Send {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String;
    async fn generate_stream(
        &self,
        prompt: &str,
        options: TextGenerationOptions,
    ) -> BoxStream<String>;
}

pub mod helpers {
    use async_stream::stream;
    use futures::{pin_mut, stream::BoxStream, Stream, StreamExt};

    pub async fn stream_to_string(s: impl Stream<Item = String>) -> String {
        pin_mut!(s);

        let mut text = "".to_owned();
        while let Some(value) = s.next().await {
            text += &value;
        }

        text
    }

    pub async fn string_to_stream(s: String) -> BoxStream<'static, String> {
        let stream = stream! {
            yield s
        };

        Box::pin(stream)
    }
}