"""Offline regression tests for timestamps in the original video's timeline.

Run with ``python -S -m unittest discover -s tests -v``. No API keys, media,
models, or third-party packages are required. Every LLM response is synthetic;
network attempts and use of the default MuAPI backend fail the test.
"""

import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest import mock

from shorts_generator import highlights


def segment(start, end, text):
    return {"start": start, "end": end, "text": text}


def candidate(title, start, end, score=80):
    return {
        "title": title,
        "start_time": start,
        "end_time": end,
        "score": score,
        "hook_sentence": title,
        "virality_reason": "Synthetic regression fixture.",
    }


class TranscriptLLM:
    """Return known absolute intervals for markers present in the transcript.

    Classification and scoring share the production callable interface. Using
    transcript markers makes the tests exercise actual chunk selection and
    prompt construction, rather than replacing the chunking implementation.
    """

    def __init__(self, responses_for_text):
        self.responses_for_text = responses_for_text
        self.classification_calls = 0
        self.scoring_texts = []

    def __call__(self, prompt):
        if prompt.startswith(highlights.CONTENT_TYPE_PROMPT):
            self.classification_calls += 1
            return json.dumps({"content_type": "commentary", "density": "high"})

        _, separator, transcript_text = prompt.partition("\n\nTranscript:\n")
        if not separator:
            raise AssertionError("Unexpected LLM task or missing transcript")
        self.scoring_texts.append(transcript_text)
        return json.dumps({"highlights": self.responses_for_text(transcript_text)})


class OfflineTimestampTests(unittest.TestCase):
    def setUp(self):
        self.forbidden_calls = []
        for target in (
            "socket.create_connection",
            "socket.getaddrinfo",
            "socket.socket.connect",
            "socket.socket.connect_ex",
            "shorts_generator.highlights.call_muapi_llm",
        ):
            patcher = mock.patch(
                target,
                side_effect=AssertionError("External calls are forbidden: " + target),
            )
            self.forbidden_calls.append(patcher.start())
            self.addCleanup(patcher.stop)

        # Keep routine application progress lines out of the unittest report.
        output_capture = redirect_stdout(io.StringIO())
        output_capture.__enter__()
        self.addCleanup(output_capture.__exit__, None, None, None)

    def tearDown(self):
        # Assert attempted calls too, including any error swallowed by a fallback.
        for forbidden_call in self.forbidden_calls:
            forbidden_call.assert_not_called()

    def test_later_chunks_preserve_absolute_times_and_all_ranked_candidates(self):
        transcript = {
            "duration": 3000.0,
            "segments": [
                segment(100.0, 160.0, "FIRST_MOMENT"),
                segment(1300.0, 1360.0, "SECOND_MOMENT"),
                segment(2500.0, 2560.0, "THIRD_MOMENT"),
            ],
        }
        before = copy.deepcopy(transcript)
        moments = {
            "FIRST_MOMENT": candidate("First", 100.0, 160.0, 70),
            "SECOND_MOMENT": candidate("Second", 1300.0, 1360.0, 95),
            "THIRD_MOMENT": candidate("Third", 2500.0, 2560.0, 85),
        }
        llm = TranscriptLLM(
            lambda text: [item for marker, item in moments.items() if marker in text]
        )

        result = highlights.get_highlights(transcript, num_clips=1, llm_fn=llm)

        self.assertEqual(
            [(item["title"], item["start_time"], item["end_time"])
             for item in result["highlights"]],
            [("Second", 1300.0, 1360.0),
             ("Third", 2500.0, 2560.0),
             ("First", 100.0, 160.0)],
        )
        self.assertEqual(llm.classification_calls, 1)
        self.assertEqual(len(llm.scoring_texts), 3)
        self.assertIn("[1300.0s] SECOND_MOMENT", llm.scoring_texts[1])
        self.assertIn("[2500.0s] THIRD_MOMENT", llm.scoring_texts[2])
        # get_highlights returns all ranked candidates; the pipeline chooses top N.
        self.assertEqual(len(result["highlights"]), 3)
        self.assertEqual(transcript, before)

    def test_threshold_changes_chunking_without_changing_absolute_results(self):
        for duration, scoring_calls in ((1799.0, 1), (1800.0, 2), (1801.0, 2)):
            with self.subTest(duration=duration):
                transcript = {
                    "duration": duration,
                    "segments": [
                        segment(100.0, 160.0, "EARLY_MOMENT"),
                        segment(1300.0, 1360.0, "LATE_MOMENT"),
                    ],
                }
                moments = {
                    "EARLY_MOMENT": candidate("Early", 100.0, 160.0, 60),
                    "LATE_MOMENT": candidate("Late", 1300.0, 1360.0, 90),
                }
                llm = TranscriptLLM(
                    lambda text: [item for marker, item in moments.items() if marker in text]
                )

                result = highlights.get_highlights(transcript, llm_fn=llm)

                self.assertEqual(
                    [(item["start_time"], item["end_time"])
                     for item in result["highlights"]],
                    [(1300.0, 1360.0), (100.0, 160.0)],
                )
                self.assertEqual(len(llm.scoring_texts), scoring_calls)
                self.assertEqual(llm.classification_calls, 1)

    def test_fractional_late_caption_timestamp_can_be_copied_without_retry(self):
        transcript = {
            "duration": 1800.0,
            "segments": [segment(1300.04, 1360.06, "FRACTIONAL_BOUNDARY")],
        }

        def copy_prompt_timestamp(text):
            # Simulate the model copying the supplied timestamp. Rounding the
            # prompt to 1300.0 would put it before the actual chunk window.
            timestamp_text, separator, _ = text.partition("s]")
            self.assertEqual(separator, "s]")
            copied_start = float(timestamp_text.removeprefix("["))
            return [candidate("Fractional", copied_start, 1360.06)]

        llm = TranscriptLLM(copy_prompt_timestamp)

        result = highlights.get_highlights(transcript, llm_fn=llm)

        self.assertEqual(
            result["highlights"], [candidate("Fractional", 1300.04, 1360.06)]
        )
        self.assertEqual(llm.classification_calls, 1)
        self.assertEqual(len(llm.scoring_texts), 1)

    def test_overlap_crossing_caption_is_complete_and_highest_score_wins(self):
        # The 1120-1280 caption was previously excluded from BOTH chunks:
        # it ended too late for the first and started too early for the second.
        transcript = {
            "duration": 1800.0,
            "segments": [
                segment(100.0, 160.0, "EARLY_CONTEXT"),
                segment(1120.0, 1280.0, "COMPLETE_BOUNDARY_STORY"),
                segment(1300.0, 1360.0, "LATE_CONTEXT"),
            ],
        }
        before = copy.deepcopy(transcript)

        def responses(text):
            result = []
            if "EARLY_CONTEXT" in text:
                result.append(candidate("Early", 100.0, 160.0, 50))
            if "LATE_CONTEXT" in text:
                result.append(candidate("Late", 1300.0, 1360.0, 70))
            if "COMPLETE_BOUNDARY_STORY" in text:
                score = 99 if "LATE_CONTEXT" in text else 65
                result.append(candidate("Boundary", 1120.0, 1280.0, score))
            return result

        llm = TranscriptLLM(responses)
        result = highlights.get_highlights(transcript, num_clips=1, llm_fn=llm)

        self.assertEqual(len(llm.scoring_texts), 2)
        self.assertTrue(all("COMPLETE_BOUNDARY_STORY" in text for text in llm.scoring_texts))
        boundary = [item for item in result["highlights"] if item["title"] == "Boundary"]
        self.assertEqual(boundary, [candidate("Boundary", 1120.0, 1280.0, 99)])
        self.assertEqual([item["score"] for item in result["highlights"]], [99, 70, 50])
        self.assertEqual(transcript, before)

    def test_chunk_metadata_keeps_video_duration_and_complete_absolute_segments(self):
        transcript = {
            "duration": 3000.0,
            "language": "ja",
            "segments": [
                segment(100.0, 160.0, "FIRST"),
                segment(1120.0, 1280.0, "CROSSING"),
                segment(1300.0, 1360.0, "SECOND"),
                segment(2500.0, 2560.0, "THIRD"),
            ],
        }
        before = copy.deepcopy(transcript)

        chunks = highlights.chunk_transcript(transcript)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(
            [[(item["start"], item["end"], item["text"]) for item in chunk["segments"]]
             for chunk in chunks],
            [[(100.0, 160.0, "FIRST"), (1120.0, 1280.0, "CROSSING")],
             [(1120.0, 1280.0, "CROSSING"), (1300.0, 1360.0, "SECOND")],
             [(2500.0, 2560.0, "THIRD")]],
        )
        self.assertEqual(
            [(chunk["_window_start"], chunk["_window_end"]) for chunk in chunks],
            [(100.0, 1280.0), (1120.0, 1360.0), (2500.0, 2560.0)],
        )
        for chunk in chunks:
            self.assertEqual(chunk["duration"], 3000.0)
            self.assertEqual(chunk["language"], "ja")
            self.assertNotIn("_offset", chunk)
        self.assertEqual(transcript, before)

    def test_short_video_keeps_valid_original_timestamps(self):
        transcript = {
            "duration": 90.0,
            "segments": [segment(0.0, 90.0, "SHORT_MOMENT")],
        }
        llm = TranscriptLLM(lambda text: [candidate("Whole clip", 0.0, 90.0, 87)])

        result = highlights.get_highlights(transcript, llm_fn=llm)

        self.assertEqual(result["highlights"], [candidate("Whole clip", 0.0, 90.0, 87)])
        self.assertEqual(llm.classification_calls, 1)
        self.assertEqual(len(llm.scoring_texts), 1)

    def test_sanitizer_accepts_exact_selected_window_edges(self):
        expected = candidate("Exact edges", 100.0, 200.0)

        result = highlights._sanitize_highlights(
            [expected], duration=300.0, window_start=100.0, window_end=200.0
        )

        self.assertEqual(result, [expected])

    def test_sanitizer_rejects_invalid_intervals_without_clamping(self):
        cases = {
            "before selected window": (99.0, 140.0),
            "after selected window": (160.0, 201.0),
            "outside both window edges": (50.0, 250.0),
            "negative start": (-1.0, 140.0),
            "reversed interval": (160.0, 140.0),
            "empty interval": (150.0, 150.0),
            "NaN start": (float("nan"), 160.0),
            "NaN end": (120.0, float("nan")),
            "infinite start": (float("inf"), 160.0),
            "infinite end": (120.0, float("inf")),
            "negative infinite start": (float("-inf"), 160.0),
            "negative infinite end": (120.0, float("-inf")),
        }
        for label, (start, end) in cases.items():
            with self.subTest(interval=label):
                result = highlights._sanitize_highlights(
                    [candidate(label, start, end)],
                    duration=300.0,
                    window_start=100.0,
                    window_end=200.0,
                )
                self.assertEqual(result, [])

    def test_sanitizer_never_extends_window_past_video_end(self):
        valid = candidate("Ends at video end", 250.0, 300.0)
        invalid = candidate("Exceeds video end", 250.0, 320.0)

        result = highlights._sanitize_highlights(
            [invalid, valid], duration=300.0, window_start=200.0, window_end=400.0
        )

        self.assertEqual(result, [valid])

    def test_sanitizer_rejects_video_overrun_without_an_explicit_window(self):
        valid = candidate("Within video", 10.0, 50.0)
        invalid = candidate("Do not truncate", 80.0, 120.0)

        result = highlights._sanitize_highlights([invalid, valid], duration=100.0)

        self.assertEqual(result, [valid])

    def test_scoring_call_passes_selected_window_to_validation(self):
        valid = candidate("Visible interval", 100.0, 200.0)
        llm = mock.Mock(return_value=json.dumps({"highlights": [
            candidate("Unseen before", 50.0, 90.0, 99),
            valid,
            candidate("Unseen after", 190.0, 220.0, 98),
        ]}))

        result = highlights.call_highlight_api(
            "[100.0s] Selected caption",
            {"content_type": "commentary", "density": "high"},
            300.0,
            num_clips=1,
            is_chunk=True,
            llm_fn=llm,
            window_start=100.0,
            window_end=200.0,
        )

        self.assertEqual(result, {"highlights": [valid]})
        llm.assert_called_once()

    def test_outside_only_scoring_response_uses_existing_bounded_retries(self):
        llm = mock.Mock(return_value=json.dumps({"highlights": [
            candidate("Do not truncate to window", 190.0, 220.0),
        ]}))

        with self.assertRaisesRegex(RuntimeError, "invalid output"):
            highlights.call_highlight_api(
                "[100.0s] Selected caption",
                {"content_type": "commentary", "density": "high"},
                300.0,
                num_clips=1,
                is_chunk=True,
                llm_fn=llm,
                window_start=100.0,
                window_end=200.0,
            )

        self.assertEqual(llm.call_count, highlights.MAX_HIGHLIGHT_API_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
