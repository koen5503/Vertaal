# STT Latency Experiment Findings

## Configuration

- **Model**: `latest_long`
- **Language**: `nl-NL`
- **Punctuation**: Enabled (`enable_automatic_punctuation=True`)
- **Silence Durations Tested**: 200ms, 400ms, 600ms, 1000ms, 2000ms, 3000ms

## Results

| Silence Gap | Segmentation Behavior | Latency Impact |
| :--- | :--- | :--- |
| **200ms - 1000ms** | **No Segmentation.** All 5 sentences were merged into one single final result at the very end of the stream. | **Catastrophic.** User sees nothing until the entire speech block ends (10s+ latency). |
| **2000ms - 3000ms** | **Partial Segmentation.** Sentences 1 & 2 were merged. Sentences 3, 4, 5 were segmented. | **High.** The first sentence had a latency of >5 seconds because it waited for the second sentence to complete before finalizing both. |

## Conclusion

**Google `latest_long` fundamentally resists segmentation for short sentences**, even with extreme silence (3 seconds).

- **The "Hack" (Phase 3) will likely fail**: Injecting 600ms of silence logic is futile because the model ignores even 1000ms-3000ms gaps when it decides to merge context.
- **Root Cause**: `latest_long` prioritizes context and punctuation over intermediate results. It holds the "buffer" open to see if the next sentence relates to the first (e.g., to decide on a comma vs period).

## Recommendations

To achieve low latency, we likely need to:

1. **Test `model='default'`**: It might be more aggressive about finalizing.
2. **Disable Punctuation**: `enable_automatic_punctuation=False` might reduce the buffer wait time.
3. **Force `single_utterance=True`**: This forces a finalized result after every pause, but requires restarting the stream (which `latest_long` might not support or might add reconnection latency).
4. **Accept Interim Results**: For live captioning, displaying `interim_results` is standard. `is_final` is only for the "committed" text. If the UI only shows "Final" text, latency will always be high with this model.
