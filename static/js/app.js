const FIELDS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"];

const FORMAT = {
  N: (v) => v,
  P: (v) => v,
  K: (v) => v,
  temperature: (v) => `${parseFloat(v).toFixed(1)}°C`,
  humidity: (v) => `${parseFloat(v).toFixed(1)}%`,
  ph: (v) => parseFloat(v).toFixed(1),
  rainfall: (v) => `${v} mm`,
};

FIELDS.forEach((f) => {
  const input = document.getElementById(f);
  const out = document.getElementById(`val-${f}`);
  input.addEventListener("input", () => {
    out.textContent = FORMAT[f](input.value);
  });
});

const form = document.getElementById("crop-form");
const submitBtn = document.getElementById("submit-btn");
const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const dialRing = document.getElementById("dial-ring");
const CIRCUMFERENCE = 264;

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = "Analyzing…";

  const payload = {};
  FIELDS.forEach((f) => (payload[f] = document.getElementById(f).value));

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || "Prediction failed");

    document.getElementById("crop-name").textContent = data.best_crop;
    document.getElementById("crop-info").textContent = data.info;
    document.getElementById("confidence-value").textContent = `${data.confidence}%`;

    const offset = CIRCUMFERENCE - (CIRCUMFERENCE * data.confidence) / 100;
    dialRing.style.strokeDashoffset = offset;

    const altList = document.getElementById("alt-list");
    altList.innerHTML = "";
    data.alternatives.forEach((alt) => {
      const chip = document.createElement("div");
      chip.className = "alt-chip";
      chip.innerHTML = `${alt.crop} <b>${alt.confidence}%</b>`;
      altList.appendChild(chip);
    });

    resultEmpty.style.display = "none";
    resultContent.classList.add("show");
  } catch (err) {
    alert(err.message || "Something went wrong. Please try again.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Analyze soil & recommend crop";
  }
});
