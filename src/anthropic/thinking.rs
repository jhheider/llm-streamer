//! Which thinking and effort controls a model accepts on the Anthropic wire.
//!
//! The Messages API has had two ways to ask for thinking, and a model rejects
//! the one it doesn't take with a 400:
//!
//! - **A fixed budget**, `thinking: {type: "enabled", budget_tokens: N}`. The
//!   only form on Claude Haiku 4.5, Sonnet 4.5, Opus 4.5 and older. Deprecated
//!   on Opus 4.6 / Sonnet 4.6, and **rejected** on Opus 4.7 and later, Sonnet
//!   5, and the Fable and Mythos models.
//! - **Adaptive**, `thinking: {type: "adaptive"}`: the model decides when and
//!   how much to think, steered by `output_config: {effort}`. Available from
//!   Opus 4.6 / Sonnet 4.6 on.
//!
//! Callers state intent (think adaptively, an effort level, a fixed budget for
//! models that need one), and [`Client`](super::Client) picks the wire form
//! each model accepts. The functions here are the table it consults; they are
//! public so an application can tell its operator when a setting won't apply.
//!
//! Model ids are matched by prefix from the first `claude-`, so dated
//! snapshots (`claude-haiku-4-5-20251001`) and gateway-prefixed ids
//! (`anthropic.claude-opus-5`, `anthropic/claude-sonnet-5`) resolve too. A
//! Claude id this table doesn't know is treated as current generation: new
//! models take adaptive thinking, and sending one a budget would 400.

use std::fmt;
use std::str::FromStr;

/// How hard the model works: `output_config: {effort}`. It governs thinking
/// depth and overall output spend. Ordered from least to most.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Effort {
    Low,
    Medium,
    High,
    /// Between `High` and `Max`. Opus 4.7 and later, Sonnet 5, Fable.
    XHigh,
    Max,
}

impl Effort {
    /// The wire value.
    pub fn as_str(self) -> &'static str {
        match self {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::XHigh => "xhigh",
            Effort::Max => "max",
        }
    }
}

impl fmt::Display for Effort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Effort {
    type Err = String;

    /// Parses the wire values, case-insensitively.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" => Ok(Effort::Low),
            "medium" => Ok(Effort::Medium),
            "high" => Ok(Effort::High),
            "xhigh" => Ok(Effort::XHigh),
            "max" => Ok(Effort::Max),
            other => Err(format!(
                "unknown effort {other:?}; expected low, medium, high, xhigh or max"
            )),
        }
    }
}

/// Which thinking request a model accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingControl {
    /// Only a fixed `budget_tokens`: Claude Haiku 4.5, Sonnet 4.5, Opus 4.5
    /// and older, and non-Claude models behind the Anthropic wire (DeepSeek's
    /// compat endpoint honours the budget form).
    Budget,
    /// Adaptive, with `budget_tokens` still accepted but deprecated: Opus 4.6
    /// and Sonnet 4.6. Adaptive is sent.
    AdaptivePreferred,
    /// Adaptive only; `budget_tokens` returns a 400: Opus 4.7 and later,
    /// Sonnet 5, Fable, Mythos, and any Claude model this table doesn't know.
    AdaptiveOnly,
}

/// The id from its first `claude-` on, or `None` for a non-Claude model.
fn claude_id(model: &str) -> Option<&str> {
    model.find("claude-").map(|i| &model[i..])
}

/// Claude models that take only a fixed thinking budget. `claude-opus-4-2`
/// and `claude-sonnet-4-2` catch the dated Claude 4.0 ids
/// (`claude-opus-4-20250514`); there is no 4.2.
const BUDGET_ONLY: &[&str] = &[
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-opus-4-1",
    "claude-opus-4-0",
    "claude-sonnet-4-0",
    "claude-opus-4-2",
    "claude-sonnet-4-2",
    "claude-3",
    "claude-2",
    "claude-instant",
];

/// Claude models that take adaptive thinking but still accept a budget.
const ADAPTIVE_PREFERRED: &[&str] = &["claude-opus-4-6", "claude-sonnet-4-6"];

/// Which thinking request `model` accepts. See [`ThinkingControl`].
pub fn thinking_control(model: &str) -> ThinkingControl {
    let Some(id) = claude_id(model) else {
        return ThinkingControl::Budget;
    };
    if BUDGET_ONLY.iter().any(|p| id.starts_with(p)) {
        ThinkingControl::Budget
    } else if ADAPTIVE_PREFERRED.iter().any(|p| id.starts_with(p)) {
        ThinkingControl::AdaptivePreferred
    } else {
        ThinkingControl::AdaptiveOnly
    }
}

/// The effort `model` will be sent when `wanted` is asked for: `wanted`
/// itself, the nearest level below it that the model supports, or `None`
/// when the model takes no effort parameter at all (Haiku 4.5, Sonnet 4.5
/// and older, where it errors; and non-Claude models, whose support is
/// unknown).
///
/// Opus 4.5 takes `low`/`medium`/`high`; Opus 4.6 and Sonnet 4.6 add `max`
/// but not `xhigh`; Opus 4.7 and later, Sonnet 5 and Fable take all five.
pub fn supported_effort(model: &str, wanted: Effort) -> Option<Effort> {
    let id = claude_id(model)?;
    match thinking_control(model) {
        ThinkingControl::AdaptiveOnly => Some(wanted),
        ThinkingControl::AdaptivePreferred => Some(match wanted {
            Effort::XHigh => Effort::High,
            other => other,
        }),
        ThinkingControl::Budget if id.starts_with("claude-opus-4-5") => {
            Some(wanted.min(Effort::High))
        }
        ThinkingControl::Budget => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_claude_models_are_adaptive_only() {
        for model in [
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-opus-5-5",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-fable-5-1",
            "claude-mythos-5-1",
            // Gateway-prefixed ids resolve the same way.
            "anthropic.claude-opus-5-5",
            "anthropic/claude-sonnet-5",
            // An id this table has never seen fails closed: no budget.
            "claude-opus-7",
        ] {
            assert_eq!(
                thinking_control(model),
                ThinkingControl::AdaptiveOnly,
                "{model}"
            );
        }
    }

    #[test]
    fn older_claude_models_and_non_claude_models_take_a_budget() {
        for model in [
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-opus-4-1",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "deepseek-v4-flash",
        ] {
            assert_eq!(thinking_control(model), ThinkingControl::Budget, "{model}");
        }
    }

    #[test]
    fn the_4_6_models_prefer_adaptive() {
        assert_eq!(
            thinking_control("claude-opus-4-6"),
            ThinkingControl::AdaptivePreferred
        );
        assert_eq!(
            thinking_control("claude-sonnet-4-6"),
            ThinkingControl::AdaptivePreferred
        );
    }

    #[test]
    fn effort_clamps_to_what_each_model_takes() {
        assert_eq!(
            supported_effort("claude-opus-5-5", Effort::XHigh),
            Some(Effort::XHigh)
        );
        assert_eq!(
            supported_effort("claude-sonnet-5", Effort::Max),
            Some(Effort::Max)
        );
        // 4.6 has max but no xhigh.
        assert_eq!(
            supported_effort("claude-opus-4-6", Effort::XHigh),
            Some(Effort::High)
        );
        assert_eq!(
            supported_effort("claude-sonnet-4-6", Effort::Max),
            Some(Effort::Max)
        );
        // Opus 4.5 tops out at high.
        assert_eq!(
            supported_effort("claude-opus-4-5", Effort::Max),
            Some(Effort::High)
        );
        assert_eq!(
            supported_effort("claude-opus-4-5", Effort::Low),
            Some(Effort::Low)
        );
        // It errors on Haiku 4.5 and Sonnet 4.5; unknown off Claude.
        assert_eq!(supported_effort("claude-haiku-4-5", Effort::Low), None);
        assert_eq!(supported_effort("claude-sonnet-4-5", Effort::High), None);
        assert_eq!(supported_effort("deepseek-v4-flash", Effort::High), None);
    }

    #[test]
    fn effort_parses_its_wire_values() {
        for e in [
            Effort::Low,
            Effort::Medium,
            Effort::High,
            Effort::XHigh,
            Effort::Max,
        ] {
            assert_eq!(e.as_str().parse::<Effort>(), Ok(e));
        }
        assert_eq!(" XHigh ".parse::<Effort>(), Ok(Effort::XHigh));
        assert!("extreme".parse::<Effort>().is_err());
    }
}
