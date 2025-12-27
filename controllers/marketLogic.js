export async function predictPrice(req, res) {
  res.json({
    crop: req.body.crop,
    price_per_quintal: 2450,
    trend: "Stable"
  });
}
