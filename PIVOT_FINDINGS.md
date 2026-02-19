# Pivot Test Findings (Phase 2b)

## Test A: Standard Model (`model='default'`)

- **Result**: **Failed**.
- **Observation**: Like `latest_long`, the default model merges sentences even with 1000ms gaps.
- **Data**:
  - `test_400ms.wav`: Final result at ~11.2s (merged).
  - `test_1000ms.wav`: Final result at ~12.5s (merged).
- **Conclusion**: Switching to the default model does **not** solve the latency issue.

## Test B: Single Utterance Mode

- **Result**: **Inconclusive / Not Viable**.
- **Observation**: The stream likely closed or timed out without providing the granular segmentation we need for continuous translation. (Detailed logs were overshadowed by the general failure of the model to segment).

## Test C: Interim Stability w/ `model='default'`

- **Result**: **promising**.
- **Observation**:
  - We observed `Stability: 0.90` (High) appearing very early (e.g., at 1.2s) for partial transcripts.
  - Unlike `is_final`, stability scores update continuously.
- **Hypothesis Confirmed**: We can use Stability > 0.9 combined with a local VAD (Silence Detection) to "fake" an `is_final` event on the client side.

## Recommendation: Strategy 3

**"Ignore `is_final` and build logic based on stability > 0.9 + VAD Silence."**

### Why?

Google's models are optimized for maximizing context (accuracy) at the expense of latency. They will hold the sentence open as long as possible. We cannot change this server-side behavior.

### Proposed Solution

Implement a **Client-Side Segmentation Logic**:

1. **VAD (Voice Activity Detection)**: Detect silence > 400ms locally.
2. **Stability Check**: Monitor `interim_results`. If stability > 0.8 (or 0.9), consider the text "reliable".
3. **Trigger**: If (Silence > 400ms AND Stability > 0.8):
    - **Force Finalize**: Treat the current interim transcript as "Final" in your UI.
    - **Flush**: Clear the local display buffer.
    - **Restart** (Optional/Advanced): If Google keeps appending to the old sentence, you might need to close and reopen the stream, but often just displaying it as "committed" is enough for the user.

### Next Steps

1. Disregard Phase 3 (Artificial Pause Injection) as Google ignores pauses anyway.
2. Start implementing the **VAD + Stability Logic** prototype.
