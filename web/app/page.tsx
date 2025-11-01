"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "../components/ui/button";
import { cn } from "../lib/utils";
import { DualLineChart, HorizontalBarChart } from "../components/charts";

type Language = "en" | "ar";
type Theme = "light" | "dark";

type Prediction = {
  churnProbability: number;
  churnLabel: number;
  userId: string;
};

const API_ENDPOINT = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000/predict";
const API_BASE = API_ENDPOINT.replace(/\/predict\/?$/i, "");
const SAMPLE_ENDPOINT = `${API_BASE}/sample`;

const metrics = [
  { id: "accuracy", value: "0.78", unit: "", labels: { en: "Accuracy", ar: "الدقة" } },
  { id: "precision", value: "0.50", unit: "", labels: { en: "Precision", ar: "الدقة الإيجابية" } },
  { id: "recall", value: "0.44", unit: "", labels: { en: "Recall", ar: "الاسترجاع" } },
  { id: "f1", value: "0.47", unit: "", labels: { en: "F1 Score", ar: "درجة F1" } },
  { id: "roc_auc", value: "0.69", unit: "", labels: { en: "ROC-AUC", ar: "مساحة ROC" } },
  { id: "pr_auc", value: "0.54", unit: "", labels: { en: "PR-AUC", ar: "مساحة PR" } },
];

const churnRateValues = {
  free: 32.0,
  paid: 18.0,
} as const;

type FeatureKey = "minutes" | "events" | "songs" | "session" | "unique";

const featureGapValues: Array<{ id: FeatureKey; value: number }> = [
  { id: "minutes", value: 280.6 },
  { id: "events", value: 69.5 },
  { id: "songs", value: 65.8 },
  { id: "session", value: 64.8 },
  { id: "unique", value: 57.6 },
] as const;

const engagementSeries = {
  retained: [
    { x: 0, y: 100 },
    { x: 1, y: 97 },
    { x: 2, y: 96 },
    { x: 3, y: 95 },
    { x: 4, y: 95 },
    { x: 5, y: 96 },
    { x: 6, y: 97 },
  ],
  churn: [
    { x: 0, y: 100 },
    { x: 1, y: 94 },
    { x: 2, y: 88 },
    { x: 3, y: 78 },
    { x: 4, y: 65 },
    { x: 5, y: 52 },
    { x: 6, y: 40 },
  ],
} as const;

const translations = {
  en: {
    languageLabel: "العربية",
    themeLight: "Light",
    themeDark: "Dark",
    heroTitle: "Customer Churn Intelligence Playbook",
    heroSubtitle:
      "End-to-end churn prediction workflow built at Thmanyah: feature engineering, Gradient Boosting baseline, MLflow tracking, FastAPI scoring, and monitoring utilities packaged for iterative growth.",
    actions: {
      tryModel: "Try the model",
      visitAnas: "Visit Anas Hamad",
    },
    metricsTitle: "Hold-out metrics",
    chartsTitle: "Visual signals",
    uploadTitle: "Score events",
    uploadDescription:
      "Upload a JSON file with an `events` array (single user) or let us sample a random user slice from the training data snapshot.",
    uploadButton: "Upload JSON",
    randomButton: "Use random data",
    uploadStatus: {
      idle: "Select a JSON payload or use random data.",
      reading: "Reading file…",
      sending: "Querying the FastAPI model…",
      success: "Prediction ready.",
      error: "Could not process this payload. Please validate the structure.",
      fallback: "Model API unreachable. Showing demo prediction instead.",
    },
    resultTitle: "Model output",
    resultLabels: {
      user: "User ID",
      probability: "Churn probability",
      bucket: "Risk bucket",
      bucketHigh: "High",
      bucketMedium: "Medium",
      bucketLow: "Low",
      recommendedAction: "Suggested action",
      actions: {
        High: "Launch a retention offer and proactive outreach.",
        Medium: "Send curated playlist recommendations and monitor engagement.",
        Low: "Keep the user in standard engagement cadence.",
      },
    },
    footer: "Built with FastAPI + scikit-learn + Next.js · API endpoint configurable via NEXT_PUBLIC_API_URL.",
    guideTitle: "JSON schema guide",
    guideSubtitle:
      "Each event mirrors a row from the activity log. Optional attributes can be omitted or set to null.",
    guideNote: "Tip: include at least 15 events across two sessions for best feature coverage.",
    guideHeaders: ["Field", "Example value", "Notes"],
    guideRows: [
      ["ts", "1538352117000", "Timestamp in milliseconds since epoch"],
      ["userId", "30", "Single user per payload"],
      ["sessionId", "29", "Integer session identifier"],
      ["page", "NextSong", "Event type"],
      ["level", "paid", "`free` or `paid`"],
      ["itemInSession", "50", "Counter within the session"],
      ["artist", "Daft Punk", "Optional string"],
      ["song", "Harder Better Faster Stronger", "Optional string"],
      ["length", "223.60", "Song duration in seconds (optional)"],
    ],
    charts: {
      churn: {
        title: "Churn by subscription level",
        caption: "Free subscribers churn at ~32% compared with ~18% for paid subscribers.",
        labels: {
          free: "Free subscribers",
          paid: "Paid subscribers",
        },
      },
      engagement: {
        title: "Engagement trajectory (indexed to 100)",
        caption: "Churners shed daily activity rapidly, while retained cohorts remain stable.",
        retainedLabel: "Retained cohort",
        churnLabel: "Churned cohort",
      },
      features: {
        title: "Engagement gap (retained – churned)",
        caption: "Positive values indicate stronger engagement among retained users over the last 30 days.",
        labels: {
          minutes: "Listening minutes (Δ)",
          events: "Events volume (Δ)",
          songs: "Songs played (Δ)",
          session: "Average session minutes (Δ)",
          unique: "Distinct songs (Δ)",
        },
      },
    },
    sections: [
      {
        id: "data",
        title: "Data compass & preparation",
        paragraphs: [
          "We operate on Sparkify-style event logs where each record is a touchpoint (song play, thumbs up, downgrade action). Datasets include the full `customer_churn.json` and a mini sample for iteration. Churn is observed when a user lands on `Cancellation Confirmation`; everyone else is retained unless rules change.",
          "Cleaning pipeline removes blank or placeholder userId values, converts timestamps from milliseconds to UTC datetimes, normalizes numerics (`sessionId`, `itemInSession`, `length`), and enforces categorical vocabularies. Identical events (same userId/session/timestamp) are deduplicated to stabilize downstream aggregations.",
        ],
        bullets: [
          "Identity keys: userId, sessionId, itemInSession.",
          "Temporal anchors: ts & registration converted from ms epoch to UTC datetimes.",
          "Stateful attributes: level (free/paid), gender, location, playback metadata.",
        ],
      },
      {
        id: "eda",
        title: "Exploratory signals",
        paragraphs: [
          "Churners skew heavily toward the free tier (≈32% churn) versus paid (≈18%). Engagement slopes downward sharply before cancellation, contrasting with steadier retained cohorts. Cancellation and downgrade pages dominate churner journeys while loyal listeners stay within song and social flows.",
        ],
        bullets: [
          "Average listening minutes drop from ≈2759 (retained) to ≈2478 (churned).",
          "Churners interact with fewer unique songs/artists and shorter sessions.",
          "Feature summary exported to docs/charts/summary.json for reproducibility.",
        ],
      },
      {
        id: "features",
        title: "Feature engineering playbook",
        paragraphs: [
          "We aggregate the last 30 days of behavior per user. For churners, events after the first cancellation confirmation are excluded to avoid leakage. Sparse users are filtered out (min 15 events & 2 sessions).",
        ],
        bullets: [
          "Volume & tempo: num_events, event_rate_per_hour, total listening minutes.",
          "Session dynamics: count, average & median duration, variability.",
          "Engagement intent: thumbs, playlists, friends, ads, downgrade gestures.",
          "Subscription state: paid event ratio, last known level, account age.",
        ],
      },
      {
        id: "modeling",
        title: "Modeling & evaluation",
        paragraphs: [
          "GradientBoostingClassifier under scikit-learn anchors the baseline with a temporal hold-out (last 14 days). Pipelines add imputation + scaling and persist via joblib while MLflow logs runs when available. Hold-out metrics captured below provide a realistic view before campaign tuning.",
        ],
        bullets: [
          "Workflow orchestrated through scripts/train.py & scripts/retrain.py.",
          "Threshold tuning (`t*`) should reflect retention economics; default at 0.5.",
          "Future upgrades: LightGBM/XGBoost comparison, probability calibration, SHAP explainability.",
        ],
      },
      {
        id: "monitoring",
        title: "Monitoring & governance",
        paragraphs: [
          "scripts/monitor.py computes PSI / KS for data drift and relative PR-AUC / recall deltas for concept drift. Thresholds are configurable in configs/monitoring.yaml. Governance guidance covers PII hashing, retention windows, and audit trails (dependencies, config, commit SHA).",
        ],
        bullets: [
          "Daily dashboard recommendation: scored volume, risk buckets, top drifting features.",
          "Retraining cadence: weekly partitions with semantic model versioning.",
          "Compliance hooks: DPIA documentation, opt-out readiness, least-privilege data access.",
        ],
      },
      {
        id: "delivery",
        title: "Packaging & handover",
        paragraphs: [
          "The repo ships with FastAPI serving (`make api`), Docker image, Makefile automation, pre-commit config, tests, and a bilingual Next.js microsite for storytelling. README and docs/TECHNICAL_REPORT.md double as implementation playbook and stakeholder brief.",
        ],
        bullets: [
          "Runbook snapshot: `make setup`, `make train`, `make monitor`, `make api`.",
          "Configs highlight: configs/training.yaml (feature window, split strategy), configs/monitoring.yaml (drift thresholds).",
          "Deliverables: artifacts/model.joblib, artifacts/metrics.json, docs/charts/ analysis kit.",
        ],
      },
    ],
  },
  ar: {
    languageLabel: "English",
    themeLight: "وضع فاتح",
    themeDark: "وضع داكن",
    heroTitle: "دليل توقع انسحاب المشتركين - ثمانية",
    heroSubtitle:
      "حل لتكليف ثمانية للتنبؤ بإنسحاب المشتركين بإستخدام نموذج تعلم آلة متقدم",
    actions: {
      tryModel: "جرّب النموذج",
      visitAnas: "موقع أنس حمد",
    },
    metricsTitle: "مقاييس مجموعة الاختبار",
    chartsTitle: "إشارات مرئية",
    metricsTitle: "مقاييس مجموعة الاختبار",
    uploadTitle: "قيّم الأحداث",
    uploadDescription:
      "يمكنك رفع ملف JSON يحتوي على مصفوفة `events` لمستخدم واحد أو استخدام عينة عشوائية من بيانات التدريب.",
    uploadButton: "رفع ملف JSON",
    randomButton: "استخدام بيانات عشوائية",
    uploadStatus: {
      idle: "اختر ملفًا أو استخدم عينة عشوائية.",
      reading: "جارٍ قراءة الملف…",
      sending: "يتم استدعاء واجهة FastAPI…",
      success: "تم استرجاع النتيجة.",
      error: "تعذر معالجة الملف. يرجى التحقق من بنيته.",
      fallback: "تعذر الوصول لواجهة النموذج. نعرض تنبؤًا تجريبيًا.",
    },
    resultTitle: "مخرجات النموذج",
    resultLabels: {
      user: "معرّف المستخدم",
      probability: "احتمال الانسحاب",
      bucket: "مستوى الخطورة",
      bucketHigh: "مرتفع",
      bucketMedium: "متوسط",
      bucketLow: "منخفض",
      recommendedAction: "الإجراء المقترح",
      actions: {
        High: "قدّم عرض ولاء وتواصل مباشرة.",
        Medium: "أرسل قوائم تشغيل مخصصة وراقب النشاط.",
        Low: "واصل حملات التفاعل المعتادة.",
      },
    },
    footer: "تم البناء باستخدام FastAPI و scikit-learn و Next.js · يمكن ضبط عنوان الواجهة عبر NEXT_PUBLIC_API_URL.",
    guideTitle: "دليل بنية JSON",
    guideSubtitle: "كل سجل يمثل حدثًا واحدًا. يمكن ترك الحقول الاختيارية فارغة أو بقيمة null.",
    guideNote: "نصيحة: يفضل توفير 15 حدثًا على الأقل موزعة على جلستين لضمان دقة التنبؤ.",
    guideHeaders: ["الحقل", "قيمة مثال", "ملاحظات"],
    guideRows: [
      ["ts", "1538352117000", "الطابع الزمني بالميلي ثانية منذ 1970"],
      ["userId", "30", "مستخدم واحد لكل طلب"],
      ["sessionId", "29", "معرّف الجلسة"],
      ["page", "NextSong", "نوع الحدث"],
      ["level", "paid", "`free` أو `paid`"],
      ["itemInSession", "50", "الترتيب داخل الجلسة"],
      ["artist", "Daft Punk", "اختياري"],
      ["song", "Harder Better Faster Stronger", "اختياري"],
      ["length", "223.60", "مدة الأغنية بالثواني (اختياري)"],
    ],
    charts: {
      churn: {
        title: "معدل الانسحاب حسب الاشتراك",
        caption: "المشتركون المجانيون ينسحبون بنسبة ≈32٪ مقابل ≈18٪ للمشتركين المدفوعين.",
        labels: {
          free: "مشتركو الخطة المجانية",
          paid: "مشتركو الخطة المدفوعة",
        },
      },
      engagement: {
        title: "مسار التفاعل (مؤشر=100)",
        caption: "يتراجع نشاط المنسحبين بسرعة بينما يحافظ الباقون على استقرار نسبي.",
        retainedLabel: "المستخدمون المحتفظ بهم",
        churnLabel: "المستخدمون المنسحبون",
      },
      features: {
        title: "فجوة التفاعل (المحتفظ بهم − المنسحبين)",
        caption: "القيم الموجبة تعكس تفوق المحتفظ بهم خلال آخر 30 يومًا.",
        labels: {
          minutes: "دقائق الاستماع (Δ)",
          events: "حجم الأحداث (Δ)",
          songs: "الأغاني المشغلة (Δ)",
          session: "متوسط مدة الجلسة (Δ)",
          unique: "الأغاني المميزة (Δ)",
        },
      },
    },
    sections: [
      {
        id: "data",
        title: "أول شيء، معالجة وتنظيف البيانات",
        paragraphs: [
          "كنت أشتغل على سجلات أحداث شبيهة بـ Sparkify، بحيث كل سجل يمثل تفاعل زي تشغيل أغنية، إعجاب، أو خفض مستوى الصوت وغيرها. البيانات كانت تشمل الملف الكامل customer_churn.json ونسخة مصغّرة للاختبارات. اعتبرت إن المستخدم ألغى اشتراكه أول ما يوصل صفحة Cancellation Confirmation، أما باقي المستخدمين فاعتبرتهم لسه موجودين إلا إذا حددت قواعد ثانية.",
          "أول شي بدأت أنظّف البيانات: شلت القيم اللي كانت فاضية أو غير صالحة، وحوّلت الطوابع الزمنية من ميلي ثانية لتوقيت UTC. بعدين شغلت الكود عشان اشوف القيم الرقمية زي الـsessionId والعناصر اللي كانت بالملف وكمان طول مدة التشغيل، وضبطت المفردات الفئوية. كمان شلت الأحداث المكررة اللي كان لها نفس المعرف والجلسة والوقت حتى أضمن إن القواعد أو بمعنى اخر (السمات) تظل ثابتة.",
        ],
        bullets: [
          "مفاتيح الهوية: userId، sessionId، itemInSession.",
          "مؤشرات الزمن: ts و registration محولة من milliseconds إلى توقيت UTC.",
          "سمات الحالة: مستوى الاشتراك (مجاني/مدفوع)، الجنس، الموقع، وبيانات التشغيل.",
        ],
      },
      {
        id: "eda",
        title: "ايش اكتشفت؟",
        paragraphs: [
          "لاحظت إن المنسحبين كانوا يميلوا للاشتراك المجاني (حوالي 32٪) مقارنة بـحوالي 18٪ من المشتركين المدفوعين. المنسحبين كمان كان عندهم تراجع واضح بالتفاعل اليومي قبل ما يلغوا، بينما المستخدمين اللي ضلّوا كانوا محافظين على نشاطهم بشكل أكثر استقرار. أغلب المنسحبين كانوا يتنقلوا بين صفحات الخفض والإلغاء، أما الباقيين فكانوا أكثر بوجودهم في صفحات الاستماع والتفاعل الاجتماعي.",
        ],
        bullets: [
          "تنخفض دقائق الاستماع من ≈2759 للمحتفظ بهم إلى ≈2478 للمنسحبين.",
          "المنسحبون يتفاعلون مع عدد أقل من الأغاني والفنانين ومع جلسات أقصر.",
          "ملخص الخصائص محفوظ في docs/charts/summary.json لضمان التتبع.",
        ],
      },
      {
        id: "features",
        title: "هندسة الخصائص",
        paragraphs: [
          "جمّعت سلوك المستخدم خلال آخر 30 يوم. بالنسبة للمنسحبين، استبعدت كل الأحداث اللي صارت بعد أول تأكيد للإلغاء حتى أتجنب تسرب البيانات. كمان استبعدت المستخدمين اللي نشاطهم كان خفيف، يعني أقل من 15 حدث وجلسيتين.",
        ],
        bullets: [
          "كثافة النشاط: عدد الأحداث، معدل الأحداث في الساعة، إجمالي دقائق الاستماع.",
          "ديناميكيات الجلسة: عدد الجلسات، متوسط ووسيط مدتها، التباين.",
          "مؤشرات التفاعل: الإعجابات، قوائم التشغيل، الأصدقاء، الإعلانات، محاولات الخفض.",
          "حالة الاشتراك: نسبة الأحداث المدفوعة، آخر مستوى معروف، عمر الحساب.",
        ],
      },
      {
        id: "modeling",
        title: "النموذج والتقييم",
        paragraphs: [
          "اعتمدت على GradientBoostingClassifier من مكتبة scikit-learn كنموذج أساسي، مع تقسيم زمني لآخر 14 يوم. بخط المعالجة، أضفت عمليات الاستكمال والتقييس، وبعدها خزّنت النموذج باستخدام joblib وسجّلت النتائج في MLflow إذا كان متوفر. المقاييس اللي طلعت من مجموعة الاختبار كانت بتمثّل الواقع قبل ما يتم ضبط حملات الاحتفاظ.",
        ],
        bullets: [
          "تنفيذ التشغيل عبر scripts/train.py و scripts/retrain.py.",
          "حدد عتبة القرار بناءً على اقتصاديات الحملات؛ القيمة الافتراضية 0.5.",
          "الترقيات المستقبلية: مقارنة LightGBM/XGBoost، معايرة الاحتمالات، وشرح SHAP.",
        ],
      },
      {
        id: "monitoring",
        title: "المراقبة والحوكمة",
        paragraphs: [
          "يشغّل scripts/monitor.py كلًا من PSI و KS لاكتشاف انحراف البيانات، ويقيس التراجع النسبي لـ PR-AUC و Recall لأنماط المفهوم. يمكن ضبط العتبات من configs/monitoring.yaml. توفر الوثائق ضوابط للهوية، الاحتفاظ بالبيانات، ومسارات التتبع (التبعيات، الإعدادات، رقم الالتزام).",
        ],
        bullets: [
          "توصية بلوحة يومية: حجم المستخدمين المقيَّمين، توزيع مستويات الخطورة، أبرز الخصائص المتغيرة.",
          "وتيرة إعادة التدريب: أسبوعيًا مع نسخ معنونة للنماذج.",
          "متطلبات الامتثال: توثيق أثر حماية البيانات، دعم خيار رفض التصنيف، صلاحيات وصول مضبوطة.",
        ],
      },
      {
        id: "delivery",
        title: "التغليف والتسليم",
        paragraphs: [
          "المستودع جاهز بواجهة FastAPI (`make api`)، صورة Docker، مهام Makefile، إعداد pre-commit، اختبارات، وموقع Next.js ثنائي اللغة لعرض القصة. يعمل README و docs/TECHNICAL_REPORT.md كدليل تنفيذ ومرجع لأصحاب المصلحة.",
        ],
        bullets: [
          "أوامر سريعة: `make setup`, `make train`, `make monitor`, `make api`.",
          "ملفات الإعداد: configs/training.yaml (نافذة الخصائص، استراتيجية التقسيم)، configs/monitoring.yaml (عتبات الانحراف).",
          "المخرجات: artifacts/model.joblib, artifacts/metrics.json, مجموعة رسوم docs/charts/.",
        ],
      },
    ],
  },
} as const;

const bucketLabel = (probability: number, language: Language) => {
  if (probability >= 0.7) {
    return language === "ar" ? translations.ar.resultLabels.bucketHigh : translations.en.resultLabels.bucketHigh;
  }
  if (probability >= 0.4) {
    return language === "ar" ? translations.ar.resultLabels.bucketMedium : translations.en.resultLabels.bucketMedium;
  }
  return language === "ar" ? translations.ar.resultLabels.bucketLow : translations.en.resultLabels.bucketLow;
};

const actionLabel = (probability: number, language: Language) => {
  const bucket = probability >= 0.7 ? "High" : probability >= 0.4 ? "Medium" : "Low";
  return translations[language].resultLabels.actions[bucket as keyof typeof translations.en.resultLabels.actions];
};

const sampleEvents = [
  {
    ts: 1538352117000,
    userId: "30",
    sessionId: 29,
    page: "NextSong",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 50,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Martha Tilston",
    song: "Rockpools",
    length: 277.89016,
  },
  {
    ts: 1538352394000,
    userId: "30",
    sessionId: 29,
    page: "NextSong",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 51,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Adam Lambert",
    song: "Time For Miracles",
    length: 282.8273,
  },
  {
    ts: 1538352676000,
    userId: "30",
    sessionId: 29,
    page: "Roll Advert",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 52,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: null,
    song: null,
    length: 0,
  },
  {
    ts: 1538352899000,
    userId: "30",
    sessionId: 29,
    page: "NextSong",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 53,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Daft Punk",
    song: "Harder Better Faster Stronger",
    length: 223.60771,
  },
  {
    ts: 1538353125000,
    userId: "30",
    sessionId: 29,
    page: "Thumbs Up",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 54,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Daft Punk",
    song: "Harder Better Faster Stronger",
    length: 223.60771,
  },
  {
    ts: 1538353300000,
    userId: "30",
    sessionId: 29,
    page: "NextSong",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 55,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Adele",
    song: "Skyfall",
    length: 285.0,
  },
  {
    ts: 1538353522000,
    userId: "30",
    sessionId: 29,
    page: "Add to Playlist",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 56,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Adele",
    song: "Skyfall",
    length: 285.0,
  },
  {
    ts: 1538353705000,
    userId: "30",
    sessionId: 29,
    page: "NextSong",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 57,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Radiohead",
    song: "Paranoid Android",
    length: 387.0,
  },
  {
    ts: 1538353940000,
    userId: "30",
    sessionId: 29,
    page: "NextSong",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 58,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Radiohead",
    song: "Karma Police",
    length: 253.0,
  },
  {
    ts: 1538354200000,
    userId: "30",
    sessionId: 29,
    page: "NextSong",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 59,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Phoenix",
    song: "1901",
    length: 211.0,
  },
  {
    ts: 1538354444000,
    userId: "30",
    sessionId: 29,
    page: "Thumbs Down",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 60,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Phoenix",
    song: "1901",
    length: 211.0,
  },
  {
    ts: 1538354666000,
    userId: "30",
    sessionId: 29,
    page: "NextSong",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 61,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Arctic Monkeys",
    song: "Do I Wanna Know?",
    length: 272.0,
  },
  {
    ts: 1538354888000,
    userId: "30",
    sessionId: 29,
    page: "NextSong",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 62,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Arctic Monkeys",
    song: "R U Mine?",
    length: 202.0,
  },
  {
    ts: 1538355150000,
    userId: "30",
    sessionId: 31,
    page: "NextSong",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 1,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Coldplay",
    song: "Paradise",
    length: 278.0,
  },
  {
    ts: 1538355400000,
    userId: "30",
    sessionId: 31,
    page: "Add Friend",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 2,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: null,
    song: null,
    length: 0,
  },
  {
    ts: 1538355600000,
    userId: "30",
    sessionId: 31,
    page: "NextSong",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 3,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Coldplay",
    song: "Adventure of a Lifetime",
    length: 263.0,
  },
  {
    ts: 1538355850000,
    userId: "30",
    sessionId: 31,
    page: "Thumbs Up",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 4,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Coldplay",
    song: "Adventure of a Lifetime",
    length: 263.0,
  },
  {
    ts: 1538356100000,
    userId: "30",
    sessionId: 31,
    page: "NextSong",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 5,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Tame Impala",
    song: "The Less I Know The Better",
    length: 216.0,
  },
  {
    ts: 1538356350000,
    userId: "30",
    sessionId: 31,
    page: "NextSong",
    auth: "Logged In",
    method: "PUT",
    status: 200,
    level: "paid",
    itemInSession: 6,
    location: "Bakersfield, CA",
    userAgent: "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    lastName: "Freeman",
    firstName: "Colin",
    registration: 1538173362000,
    gender: "M",
    artist: "Tame Impala",
    song: "Let It Happen",
    length: 301.0,
  },
] as const;

export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const scoringRef = useRef<HTMLDivElement | null>(null);
  const language: Language = "ar";
  const [theme, setTheme] = useState<Theme>("light");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [status, setStatus] = useState<string>(translations.ar.uploadStatus.idle);
  const [isUploading, setIsUploading] = useState(false);
  const [showGuide, setShowGuide] = useState(true);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    document.documentElement.lang = language === "ar" ? "ar" : "en";
    document.body.dir = language === "ar" ? "rtl" : "ltr";
    setStatus(translations[language].uploadStatus.idle);
  }, [language]);

  const copy = translations[language];

  const scrollToScoring = () => {
    scoringRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const sendPayload = async (payload: any, statusMessages: (typeof translations)["en"]["uploadStatus"]) => {
    setIsUploading(true);
    setStatus(statusMessages.sending);
    try {
      const response = await fetch(API_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = await response.json();
      setPrediction({
        churnProbability: data.churn_probability ?? data.churnProbability ?? 0,
        churnLabel: data.churn_label ?? data.churnLabel ?? 0,
        userId: data.user_id ?? data.userId ?? payload.events?.[0]?.userId ?? "---",
      });
      setStatus(statusMessages.success);
    } catch (error) {
      console.error("Scoring failed", error);
      setPrediction({
        churnProbability: 0.37,
        churnLabel: 0,
        userId: payload?.events?.[0]?.userId ?? "demo",
      });
      setStatus(statusMessages.fallback ?? statusMessages.error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setStatus(copy.uploadStatus.reading);

    try {
      const text = await file.text();
      let parsed = JSON.parse(text);
      if (!parsed) throw new Error("Empty payload");
      if (Array.isArray(parsed)) parsed = { events: parsed };
      if (!parsed.events || !Array.isArray(parsed.events)) {
        throw new Error("Missing events array");
      }
      await sendPayload(parsed, copy.uploadStatus);
    } catch (error) {
      console.error("Upload failed:", error);
      setPrediction(null);
      setStatus(copy.uploadStatus.error);
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleRandomData = async () => {
    try {
      const response = await fetch(SAMPLE_ENDPOINT);
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const sample = await response.json();
      await sendPayload({ events: sample.events }, copy.uploadStatus);
    } catch (error) {
      console.error("Failed to fetch sample events", error);
      await sendPayload({ events: sampleEvents }, copy.uploadStatus);
    }
  };

  const handleUploadButton = () => {
    setShowGuide(true);
    fileInputRef.current?.click();
  };

  return (
    <main className={language === "ar" ? "font-ar" : "font-en"}>
      <div className="toolbar">
        <Button variant="outline" size="sm" onClick={() => setTheme(theme === "light" ? "dark" : "light")}>
          {theme === "light" ? copy.themeDark : copy.themeLight}
        </Button>
      </div>

      <article className={cn("article")}>
        <header>
          <h1>{copy.heroTitle}</h1>
          <p className="lead">{copy.heroSubtitle}</p>
          <div className="actions">
            <Button onClick={scrollToScoring}>{copy.actions.tryModel}</Button>
            <Button
              variant="outline"
              onClick={() => window.open("https://www.anashamad.site/ar", "_blank", "noopener,noreferrer")}
            >
              {copy.actions.visitAnas}
            </Button>
          </div>
        </header>

        <section>
          <h2>{copy.metricsTitle}</h2>
          <div className="metrics-inline">
            {metrics.map((metric) => (
              <div key={metric.id}>
                <span>{metric.labels[language]}</span>
                <strong>
                  {metric.value}
                  {metric.unit}
                </strong>
              </div>
            ))}
          </div>
        </section>

        <section>
          <h2>{copy.chartsTitle}</h2>
          <div className="chart-group">
            <HorizontalBarChart
              title={copy.charts.churn.title}
              caption={copy.charts.churn.caption}
              direction={language === "ar" ? "rtl" : "ltr"}
              data={(["free", "paid"] as const).map((tier) => ({
                id: tier,
                label: copy.charts.churn.labels[tier],
                value: churnRateValues[tier],
                emphasis: tier === "free",
              }))}
            />
            <DualLineChart
              title={copy.charts.engagement.title}
              caption={copy.charts.engagement.caption}
              retainedLabel={copy.charts.engagement.retainedLabel}
              churnLabel={copy.charts.engagement.churnLabel}
              direction={language === "ar" ? "rtl" : "ltr"}
              retainedSeries={engagementSeries.retained}
              churnSeries={engagementSeries.churn}
            />
            <HorizontalBarChart
              title={copy.charts.features.title}
              caption={copy.charts.features.caption}
              direction={language === "ar" ? "rtl" : "ltr"}
              data={featureGapValues.map((item) => ({
                id: item.id,
                label: copy.charts.features.labels[item.id],
                value: item.value,
                emphasis: item.id === "minutes",
              }))}
            />
          </div>
        </section>

        {copy.sections.map((section) => (
          <section key={section.id}>
            <h2>{section.title}</h2>
            {section.paragraphs.map((paragraph, idx) => (
              <p key={idx} style={{ margin: "0 0 0.9rem 0" }}>
                {paragraph}
              </p>
            ))}
            {section.bullets && (
              <ul style={{ margin: "0.4rem 0 0 0", paddingInlineStart: language === "ar" ? "1.1rem" : "1.4rem" }}>
                {section.bullets.map((item) => (
                  <li key={item} style={{ marginBottom: "0.35rem" }}>
                    {item}
                  </li>
                ))}
              </ul>
            )}
          </section>
        ))}

        <section ref={scoringRef}>
          <h2>{copy.uploadTitle}</h2>
          <p>{copy.uploadDescription}</p>
          <div className="actions" style={{ marginTop: "1rem", marginBottom: "1rem" }}>
            <Button onClick={handleRandomData} disabled={isUploading}>
              {copy.randomButton}
            </Button>
            <Button variant="outline" onClick={handleUploadButton} disabled={isUploading}>
              {copy.uploadButton}
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept="application/json"
              onChange={handleFileChange}
              aria-label="Upload JSON file"
              style={{ display: "none" }}
            />
          </div>

          {showGuide && (
            <div className={cn("guide-card", language === "ar" && "rtl")}>
              <strong>{copy.guideTitle}</strong>
              <p className="guide-subtitle">{copy.guideSubtitle}</p>
              <table className="guide-table">
                <thead>
                  <tr>
                    {copy.guideHeaders.map((header) => (
                      <th key={header}>{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {copy.guideRows.map(([field, example, note]) => (
                    <tr key={field}>
                      <td>{field}</td>
                      <td>{example}</td>
                      <td>{note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="guide-note">{copy.guideNote}</p>
            </div>
          )}

          <div className="scoring-panel">
            <p className="status-text">{status}</p>
            {prediction && (
              <div className="prediction">
                <div className="prediction-item">
                  <span>{copy.resultLabels.user}</span>
                  <strong>{prediction.userId}</strong>
                </div>
                <div className="prediction-item">
                  <span>{copy.resultLabels.probability}</span>
                  <strong>{(prediction.churnProbability * 100).toFixed(1)}%</strong>
                </div>
                <div className="prediction-item">
                  <span>{copy.resultLabels.bucket}</span>
                  <strong>{bucketLabel(prediction.churnProbability, language)}</strong>
                </div>
                <div className="prediction-item">
                  <span>{copy.resultLabels.recommendedAction}</span>
                  <strong style={{ fontSize: "1rem", lineHeight: 1.5 }}>
                    {actionLabel(prediction.churnProbability, language)}
                  </strong>
                </div>
              </div>
            )}
          </div>
        </section>

      </article>
    </main>
  );
}
