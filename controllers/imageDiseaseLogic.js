export async function detectDiseaseFromImage(req, res) {
  const { filename, crop } = req.body;

  res.json({
    crop,
    disease: "Powdery Mildew",
    confidence: 0.81,
    recommendation: "Apply sulfur-based fungicide"
  });
}
