"""
InterviewAnalyzer — Core NLP pipeline for AI User Interview Analyzer
Extracts themes, pain points, and opportunities from raw interview transcripts.
"""

import re
import nltk
import string
from collections import Counter

import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Download required NLTK data (silent)
try:
      nltk.data.find("tokenizers/punkt")
except LookupError:
      nltk.download("punkt", quiet=True)

try:
      nltk.data.find("corpora/stopwords")
except LookupError:
      nltk.download("stopwords", quiet=True)

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

# Pain-related signal words for keyword heuristics
PAIN_KEYWORDS = {
      "difficult", "hard", "confusing", "broken", "slow", "crash", "bug",
      "frustrating", "annoying", "hate", "terrible", "awful", "broken",
      "impossible", "can't", "cannot", "won't", "doesn't", "problem",
      "issue", "fail", "error", "useless", "painful", "struggle",
}

# Positive signal words
POSITIVE_KEYWORDS = {
      "love", "great", "excellent", "perfect", "easy", "fast", "intuitive",
      "helpful", "useful", "amazing", "fantastic", "best", "clear", "smooth",
      "enjoy", "like", "recommend", "brilliant", "wonderful",
}


class InterviewAnalyzer:
      """
          NLP pipeline that analyzes user interview transcripts to extract:
              - Recurring themes (unsupervised clustering)
                  - Pain points (negative sentiment + pain keywords)
                      - Positive signals / opportunities
                          - PM-ready summary report
                              """

    def __init__(self, n_themes: int = 5, pain_threshold: float = -0.2):
              self.n_themes = n_themes
              self.pain_threshold = pain_threshold

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, text: str) -> dict:
              """
                      Main analysis pipeline.

                              Args:
                                          text: Raw interview transcript string

                                                  Returns:
                                                              dict with keys: themes, pain_points, opportunities,
                                                                                          pm_summary, sentence_count, theme_count
                                                                                                  """
              sentences = self._segment(text)
              if len(sentences) < self.n_themes:
                            # Fallback: reduce themes if transcript is short
                            self.n_themes = max(2, len(sentences) // 2)

              scored = self._score_sentences(sentences)
              clustered = self._cluster_themes(scored)
              themes = self._summarize_themes(clustered)
              pain_points = self._extract_pain_points(clustered)
              opportunities = self._extract_opportunities(clustered)
              pm_summary = self._generate_pm_summary(themes, pain_points, opportunities)

        return {
                      "themes": themes,
                      "pain_points": pain_points,
                      "opportunities": opportunities,
                      "pm_summary": pm_summary,
                      "sentence_count": len(sentences),
                      "theme_count": len(themes),
        }

    # ── Private Methods ────────────────────────────────────────────────────────

    def _segment(self, text: str) -> list[str]:
              """Split transcript into clean, non-empty sentences."""
              raw = sent_tokenize(text)
              sentences = []
              for s in raw:
                            s = s.strip()
                            # Remove speaker labels (e.g., "I:", "P:", "Interviewer:")
                            s = re.sub(r"^[A-Za-z]+\s*:\s*", "", s)
                            s = s.strip()
                            if len(s.split()) >= 4:  # Ignore very short fragments
                                sentences.append(s)
                                      return sentences

    def _score_sentences(self, sentences: list[str]) -> list[dict]:
              """Attach sentiment score and keyword flags to each sentence."""
              scored = []
              for s in sentences:
                            blob = TextBlob(s)
                            sentiment = blob.sentiment.polarity
                            words = set(s.lower().split())
                            has_pain = bool(words & PAIN_KEYWORDS)
                            has_positive = bool(words & POSITIVE_KEYWORDS)
                            scored.append({
                                "sentence": s,
                                "sentiment": round(sentiment, 4),
                                "has_pain": has_pain,
                                "has_positive": has_positive,
                            })
                        return scored

    def _cluster_themes(self, scored: list[dict]) -> list[dict]:
              """Assign each sentence to a theme cluster using KMeans + TF-IDF."""
        sentences = [item["sentence"] for item in scored]

        vectorizer = TfidfVectorizer(
                      stop_words="english",
                      max_features=500,
                      ngram_range=(1, 2),
        )
        X = vectorizer.fit_transform(sentences)

        kmeans = KMeans(
                      n_clusters=self.n_themes,
                      random_state=42,
                      n_init=10,
        )
        labels = kmeans.fit_predict(X)

        # Get top terms per cluster for theme naming
        feature_names = vectorizer.get_feature_names_out()
        cluster_top_terms = {}
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        for i in range(self.n_themes):
                      top_terms = [feature_names[ind] for ind in order_centroids[i, :3]]
                      cluster_top_terms[i] = " / ".join(top_terms)

        for i, item in enumerate(scored):
                      cluster_id = int(labels[i])
                      item["cluster"] = cluster_id
                      item["theme_label"] = f"Theme {cluster_id + 1}: {cluster_top_terms[cluster_id]}"

        return scored

    def _summarize_themes(self, clustered: list[dict]) -> list[dict]:
              """Aggregate theme-level stats: frequency, avg sentiment, top quote."""
        theme_map = {}
        for item in clustered:
                      tid = item["cluster"]
                      if tid not in theme_map:
                                        theme_map[tid] = {
                                                              "theme_id": tid,
                                                              "theme_label": item["theme_label"],
                                                              "sentences": [],
                                                              "sentiments": [],
                                        }
                                    theme_map[tid]["sentences"].append(item["sentence"])
            theme_map[tid]["sentiments"].append(item["sentiment"])

        themes = []
        for tid, data in sorted(theme_map.items()):
                      avg_sentiment = round(float(np.mean(data["sentiments"])), 3)
            # Pick the sentence closest to the mean sentiment as top quote
            diffs = [abs(s - avg_sentiment) for s in data["sentiments"]]
            top_quote = data["sentences"][int(np.argmin(diffs))]
            themes.append({
                              "theme_label": data["theme_label"],
                              "frequency": len(data["sentences"]),
                              "avg_sentiment": avg_sentiment,
                              "signal": (
                                                    "🔴 Pain" if avg_sentiment < -0.2
                                                    else "🟢 Positive" if avg_sentiment > 0.2
                                                    else "🟡 Neutral"
                              ),
                              "top_quote": f'"{top_quote[:120]}..."' if len(top_quote) > 120 else f'"{top_quote}"',
            })

        # Sort by frequency descending
        return sorted(themes, key=lambda x: x["frequency"], reverse=True)

    def _extract_pain_points(self, clustered: list[dict]) -> list[dict]:
              """Return sentences with strong negative sentiment or pain keywords."""
        pain_points = [
                      {
                                        "sentence": item["sentence"],
                                        "sentiment": item["sentiment"],
                                        "theme": item["theme_label"],
                      }
                      for item in clustered
                      if item["sentiment"] <= self.pain_threshold or item["has_pain"]
        ]
        return sorted(pain_points, key=lambda x: x["sentiment"])[:10]

    def _extract_opportunities(self, clustered: list[dict]) -> list[dict]:
              """Return sentences with strong positive sentiment or positive keywords."""
        opportunities = [
                      {
                                        "sentence": item["sentence"],
                                        "sentiment": item["sentiment"],
                                        "theme": item["theme_label"],
                      }
                      for item in clustered
                      if item["sentiment"] >= 0.2 or item["has_positive"]
        ]
        return sorted(opportunities, key=lambda x: x["sentiment"], reverse=True)[:10]

    def _generate_pm_summary(
              self,
              themes: list[dict],
              pain_points: list[dict],
              opportunities: list[dict],
    ) -> str:
              """Generate a plain-text PM summary report."""
        lines = ["=" * 60]
        lines.append("PM INSIGHT SUMMARY — AI User Interview Analyzer")
        lines.append("=" * 60)
        lines.append(f"\nThemes extracted: {len(themes)}")
        lines.append(f"Pain signals: {len(pain_points)}")
        lines.append(f"Positive signals: {len(opportunities)}\n")

        lines.append("── TOP THEMES ───────────────────────────────────────────")
        for i, t in enumerate(themes[:5], 1):
                      lines.append(
                          f"{i}. {t['theme_label']} | {t['frequency']} mentions | "
                          f"Sentiment: {t['avg_sentiment']} {t['signal']}"
        )
            lines.append(f"   Quote: {t['top_quote']}")

        if pain_points:
                      lines.append("\n── TOP PAIN POINTS ──────────────────────────────────────")
            for i, p in enumerate(pain_points[:3], 1):
                              lines.append(f"{i}. [{p['sentiment']:.2f}] \"{p['sentence'][:100]}\"")

        if opportunities:
                      lines.append("\n── TOP OPPORTUNITIES ────────────────────────────────────")
            for i, o in enumerate(opportunities[:3], 1):
                              lines.append(f"{i}. [{o['sentiment']:.2f}] \"{o['sentence'][:100]}\"")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
