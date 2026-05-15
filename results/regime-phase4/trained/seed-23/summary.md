# Phase 4 Detector Swap Seed 23

- Train rows: 166556
- Test rows: 41640
- Trained terminals: svm_regime_is_trend, svm_regime_is_range, svm_regime_is_neutral, svm_regime_is_fallback, svm_regime_is_no_trade, svm_regime_is_empty
- Handoff parity match rate: 1.000000
- Handoff classifier SHA-256: 622e4dc475e4323d26ca63ba5c2bbdcd784719244113eb3a2c88f657dfc7ac40
- Terminal counts: {'svm_regime_is_trend': 3709, 'svm_regime_is_range': 0, 'svm_regime_is_neutral': 37931, 'svm_regime_is_fallback': 0, 'svm_regime_is_no_trade': 0, 'svm_regime_is_empty': 0}

Note: this artifact verifies trained-detector materialization and trusted handoff loading. Full selected-source evolution summaries are written by the liq-evolution CLI under the selected detector-source output directory; explicit evolved-vs-trained paired comparison artifacts require `--phase4-paired-comparison`.
