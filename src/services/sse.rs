//! Minimal Server-Sent Events line parser for OpenAI-compatible streams.
//!
//! Bytes arrive in arbitrary chunks (a JSON payload or even a multi-byte character can be
//! split across two reads), so the parser only decodes complete lines.

#[derive(Default)]
pub struct SseParser {
    buf: Vec<u8>,
}

impl SseParser {
    /// Feed raw bytes; returns the payload of every complete `data:` line.
    /// Comments (`: keep-alive`), `event:`/`id:` fields and blank lines are ignored.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\n', '\r']);
            if let Some(rest) = line.strip_prefix("data:") {
                out.push(rest.trim_start().to_string());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_complete_events_and_ignores_noise() {
        let mut p = SseParser::default();
        let got = p.push(b": keep-alive\n\nevent: x\ndata: {\"a\":1}\n\ndata: [DONE]\n\n");
        assert_eq!(got, vec!["{\"a\":1}", "[DONE]"]);
    }

    #[test]
    fn handles_any_chunk_boundary_including_inside_a_utf8_character() {
        let stream = "data: {\"t\":\"é 漢\"}\r\n\r\ndata: [DONE]\r\n\r\n".as_bytes();
        for cut in 1..stream.len() {
            let mut p = SseParser::default();
            let mut got = p.push(&stream[..cut]);
            got.extend(p.push(&stream[cut..]));
            assert_eq!(got, vec!["{\"t\":\"é 漢\"}", "[DONE]"], "split at byte {cut}");
        }
    }

    #[test]
    fn a_partial_line_waits_for_its_newline() {
        let mut p = SseParser::default();
        assert!(p.push(b"data: par").is_empty());
        assert_eq!(p.push(b"tial\n"), vec!["partial"]);
    }
}
