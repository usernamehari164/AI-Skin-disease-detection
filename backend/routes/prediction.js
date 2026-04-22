const express = require("express");
const multer = require("multer");
const { execFile } = require("child_process");
const path = require("path");
const fs = require("fs");
const auth = require("../middleware/auth");
const Prediction = require("../models/Prediction");

const router = express.Router();

// MULTER CONFIG
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const userDir = path.join("uploads", req.user.id);
        if (!fs.existsSync(userDir)) {
            fs.mkdirSync(userDir, { recursive: true });
        }
        cb(null, userDir);
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname));
    }
});

const upload = multer({ storage: storage });

// @route   POST api/predict
// @desc    Predict skin disease from image
router.post("/", auth, upload.single("image"), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
    }

    const imagePath = req.file.path;
    const scriptPath = path.join(__dirname, "../../predict.py");

    execFile(
        "python",
        [scriptPath, imagePath],
        async (error, stdout, stderr) => {
            if (error) {
                console.error("Exec Error:", stderr);
                return res.status(500).json({
                    error: "Prediction failed",
                    details: stderr
                });
            }

            try {
                const result = JSON.parse(stdout);
                
                // LOAD ADVICE FROM JSON
                const advicePath = path.join(__dirname, "../../model/disease_advice.json");
                let diseaseData = {
                    advice: "Please consult a professional dermatologist for a formal diagnosis and treatment plan.",
                    severity: "medium",
                    urgency: "Consultation Recommended"
                };

                if (fs.existsSync(advicePath)) {
                    const adviceJson = JSON.parse(fs.readFileSync(advicePath, "utf8"));
                    if (adviceJson[result.prediction]) {
                        diseaseData = adviceJson[result.prediction];
                    }
                }

                // SAVE TO DATABASE
                const prediction = new Prediction({
                    userId: req.user.id,
                    diseaseName: result.prediction, 
                    confidence: result.confidence,
                    advice: diseaseData.advice,
                    severity: diseaseData.severity,
                    urgency: diseaseData.urgency,
                    imagePath: req.file.filename
                });

                await prediction.save();
                res.json(prediction);
            } catch (e) {
                console.error("JSON Parse Error:", stdout);
                res.status(500).json({
                    error: "Invalid response from AI model",
                    raw: stdout
                });
            }
        }
    );
});

// @route   GET api/predict/history
// @desc    Get user scan history
router.get('/history', auth, async (req, res) => {
    try {
        const history = await Prediction.find({ userId: req.user.id }).sort({ createdAt: -1 });
        res.json(history);
    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server Error');
    }
});

// @route   GET api/predict/:id
// @desc    Get specific prediction
router.get('/:id', auth, async (req, res) => {
    try {
        const prediction = await Prediction.findById(req.params.id);
        if (!prediction) return res.status(404).json({ msg: 'Prediction not found' });
        if (prediction.userId.toString() !== req.user.id) return res.status(401).json({ msg: 'Not authorized' });
        res.json(prediction);
    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server Error');
    }
});

// @route   DELETE api/predict/:id
// @desc    Delete a prediction
router.delete('/:id', auth, async (req, res) => {
    try {
        console.log(`Attempting to delete scan: ${req.params.id} for user: ${req.user.id}`);
        
        const prediction = await Prediction.findById(req.params.id);

        if (!prediction) {
            console.error('Prediction not found');
            return res.status(404).json({ msg: 'Prediction not found' });
        }

        // Check user
        if (prediction.userId.toString() !== req.user.id) {
            console.error('Unauthorized delete attempt');
            return res.status(401).json({ msg: 'User not authorized' });
        }

        // Delete image file (Absolute path) - Wrap in try-catch so missing file doesn't break DB delete
        try {
            const fullImagePath = path.join(__dirname, "../uploads", req.user.id, prediction.imagePath);
            if (fs.existsSync(fullImagePath)) {
                fs.unlinkSync(fullImagePath);
            }
        } catch (fileErr) {
            console.warn('File deletion failed (may already be gone):', fileErr.message);
        }

        await Prediction.findByIdAndDelete(req.params.id);

        res.json({ msg: 'Prediction removed' });
    } catch (err) {
        console.error('Server error during deletion:', err.message);
        res.status(500).send('Server Error');
    }
});

module.exports = router;
