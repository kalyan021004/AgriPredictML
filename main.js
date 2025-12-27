import express from "express";
import cors from "cors";

import { recommendCrop } from "./controllers/cropLogic.js";
import { detectDisease } from "./controllers/diseaseLogic.js";
import { predictPrice } from "./controllers/marketLogic.js";
import { detectDiseaseFromImage } from "./controllers/imageDiseaseLogic.js";

const app = express();
const PORT = process.env.PORT || 6001;

app.use(cors());
app.use(express.json());

/* =========================
   HEALTH CHECK
========================= */
app.get("/health", (req, res) => {
  res.json({ status: "ML service running" });
});

/* =========================
   ML ENDPOINTS
========================= */
app.post("/api/ml/crop-recommendation", recommendCrop);
app.post("/api/ml/disease-detect", detectDisease);
app.post("/api/ml/market-price", predictPrice);
app.post("/api/ml/disease-from-image", detectDiseaseFromImage);

app.listen(PORT, () =>
  console.log(`🚀 ML Service running on http://localhost:${PORT}`)
);
