import unittest
from score import score
import joblib
import subprocess
import os
import requests
import time
import signal

class TestScoreFunction(unittest.TestCase):
    def setUp(self):
        # Load the trained model
        model = joblib.load('trained_logistic_regression.pkl')
        self.model = model

    def test_smoke(self):
        text = "Sample text"
        threshold = 0.5
        prediction, propensity = score(text, self.model, threshold)
        self.assertIsNotNone(prediction)
        self.assertIsNotNone(propensity)

    def test_format(self):
        text = "Sample text"
        threshold = 0.5
        prediction, propensity = score(text, self.model, threshold)
        self.assertIsInstance(prediction, bool)
        self.assertIsInstance(propensity, float)

    def test_prediction_value(self):
        text = "Sample text"
        threshold = 0.5
        prediction, _ = score(text, self.model, threshold)
        self.assertIn(prediction, [True, False])

    def test_propensity_score(self):
        text = "Sample text"
        threshold = 0.5
        _, propensity = score(text, self.model, threshold)
        self.assertGreaterEqual(propensity, 0)
        self.assertLessEqual(propensity, 1)

    def test_threshold_zero(self):
        text = "Sample text"
        prediction, _ = score(text, self.model, 0)
        self.assertTrue(prediction)

    def test_threshold_one(self):
        text = "Sample text"
        prediction, _ = score(text, self.model, 1)
        self.assertFalse(prediction)

    def test_obvious_spam(self):
        spam_text = "Get rich quick! Win 1 million dollars with a single click, along with a free trip to Hawaii!! Check it out asap!!!"
        threshold = 0.5
        spam_prediction, _ = score(spam_text, self.model, threshold)
        self.assertTrue(spam_prediction)

    def test_obvious_non_spam(self):
        non_spam_text = "Hello, how are you today?"
        threshold = 0.5
        non_spam_prediction, _ = score(non_spam_text, self.model, threshold)
        self.assertFalse(non_spam_prediction)


class TestFlaskApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        
        os.environ['FLASK_APP'] = 'app'
        os.environ['FLASK_RUN_PORT'] = '5000'
        os.environ['FLASK_ENV'] = 'development'
        # Start the Flask app in a separate process
        cls.flask_process = subprocess.Popen(["flask", "run"])
        # Wait for the Flask app to start
        time.sleep(3)

    @classmethod
    def tearDownClass(cls):
        # Terminate the Flask app process by sending a SIGINT signal to the subprocess
        # cls.flask_process.terominat()
        # cls.flask_process.wait()
        os.kill(cls.flask_process.pid, signal.SIGINT)

    def test_flask_app(self):
        """Test the Flask endpoint."""
        response = requests.post('http://localhost:5000/score', json={'text': 'This is AML HW 3', 'threshold': 0.5})
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json())
        self.assertIn('propensity', response.json())


if __name__ == '__main__':
    unittest.main()
