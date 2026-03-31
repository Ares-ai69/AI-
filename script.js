let pipeline = null;

async function initAI() {
  pipeline = await window.Transformers.pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
}
initAI();

async function runAI() {
  const text = document.getElementById("text").value;
  const minScore = parseFloat(document.getElementById("score").value);
  const maxKeywords = parseInt(document.getElementById("max").value);

  if (!text || !pipeline) return;
  document.getElementById("result").innerText = "处理中...";

  const result = await pipeline(text, { pooling: "mean" });
  const words = text.split(/\W+/).filter(w => w.length > 1);
  document.getElementById("result").innerText = words.slice(0, maxKeywords).join("，");
}
