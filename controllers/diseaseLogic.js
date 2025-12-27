export async function detectDisease(req, res) {
  res.json({
    disease: "Leaf Blight",
    confidence: 0.76,
    treatment: "Use fungicide and ensure proper drainage"
  });
}
