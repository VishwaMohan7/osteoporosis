import os
os.environ['GEMINI_API_KEY'] = 'AIzaSyBDxhCw6eCS6staVFARHK9XMubRm3lUJWE'
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from skimage import morphology, measure
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F

# Load environment variables for API key
from dotenv import load_dotenv
import os
import requests
load_dotenv()  # Load variables from .env file

# Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)

# Define FeatureClassifier
class FeatureClassifier(nn.Module):
    def __init__(self, input_size=100):
        super(FeatureClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)

# Custom Dataset for testing
class XrayTestDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Failed to load image at {self.image_paths[idx]}")
            image = np.zeros(IMG_SIZE, dtype=np.uint8)
        image = cv2.resize(image, IMG_SIZE)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# OsteoporosisEarlyDetectionSystem for inference
class OsteoporosisEarlyDetectionSystem:
    def __init__(self, model_path):
        self.model = FeatureClassifier(input_size=100).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()  # Set to evaluation mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
        ])
        self.risk_thresholds = {
            'cortical_thinning': 0.4,
            'trabecular_degradation': 0.35,
            'radiolucency': 0.3,
            'compression_fractures': 0.8,
            'endplate_irregularities': 0.5,
            'geometry_alteration': 0.45
        }

    def preprocess_xray(self, image):
        # Handle tensor input or multi-dimensional arrays
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        # Squeeze unnecessary dimensions
        while len(image.shape) > 2 and image.shape[-1] <= 1:
            image = image.squeeze(-1)
        if len(image.shape) == 3 and image.shape[0] == 1:  # [1, H, W] or [1, C, H, W]
            image = image.squeeze(0)  # Remove batch dimension
        elif len(image.shape) == 3 and image.shape[-1] in [1, 3]:  # [H, W, C]
            image = image.squeeze(axis=-1) if image.shape[-1] == 1 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) > 2:  # Unexpected shape, try to extract 2D array
            image = image[0] if image.shape[0] == 1 else image  # Take first batch if present
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced

    def segment_bone(self, image):
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bone_mask = np.zeros_like(image, dtype=np.uint8)
        for contour in contours:
            if 5000 < cv2.contourArea(contour) < image.size * 0.9:
                cv2.drawContours(bone_mask, [contour], -1, 255, -1)
        return bone_mask

    def analyze_cortical_thinning(self, image):
        enhanced = self.preprocess_xray(image)
        bone_mask = self.segment_bone(enhanced)
        edges = cv2.Canny(enhanced, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=bone_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cortical_measurements = []
        cortical_mask = np.zeros_like(enhanced)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(cortical_mask, [contour], -1, 255, -1)
                thickness_measurements = self._measure_cortical_thickness(contour, enhanced)
                cortical_measurements.extend(thickness_measurements)
        if cortical_measurements:
            avg_thickness = np.mean(cortical_measurements)
            thickness_variation = np.std(cortical_measurements)
            normal_thickness_range = (0.05, 0.15)
            thinning_score = max(0, 1 - (avg_thickness / normal_thickness_range[1]))
            variation_penalty = min(1, thickness_variation * 2)
            cortical_risk = min(1.0, thinning_score + variation_penalty * 0.3)
        else:
            cortical_risk = 0.5
        return cortical_risk, cortical_measurements, cortical_mask

    def _measure_cortical_thickness(self, contour, image):
        measurements = []
        contour_points = contour.reshape(-1, 2)
        sample_indices = np.linspace(0, len(contour_points)-1, min(25, len(contour_points)), dtype=int)
        for idx in sample_indices:
            point = contour_points[idx]
            x, y = point
            if idx > 0 and idx < len(contour_points) - 1:
                prev_point = contour_points[idx-1]
                next_point = contour_points[idx+1]
                tangent = next_point - prev_point
                normal = np.array([-tangent[1], tangent[0]])
                normal = normal / (np.linalg.norm(normal) + 1e-6)
                thickness = self._sample_thickness_along_normal(image, point, normal, max_length=20)
                if thickness > 0:
                    measurements.append(thickness / max(image.shape))
        return measurements

    def _sample_thickness_along_normal(self, image, center, normal, max_length=20):
        h, w = image.shape
        thickness = 0
        for direction in [1, -1]:
            for distance in range(1, max_length):
                sample_point = center + direction * distance * normal
                x, y = int(sample_point[0]), int(sample_point[1])
                if 0 <= x < w and 0 <= y < h:
                    if image[y, x] < 100:
                        thickness = distance
                        break
                else:
                    break
        return thickness

    def analyze_trabecular_bone(self, image):
        enhanced = self.preprocess_xray(image)
        bone_mask = self.segment_bone(enhanced)
        center_region = cv2.bitwise_and(enhanced, enhanced, mask=bone_mask)[enhanced.shape[0]//4:3*enhanced.shape[0]//4, enhanced.shape[1]//4:3*enhanced.shape[1]//4]
        def calculate_lbp_variance(image, radius=2):
            rows, cols = image.shape
            lbp_values = []
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = image[i, j]
                    binary_pattern = 0
                    points = [(-radius, -radius), (-radius, 0), (-radius, radius), (0, radius), (radius, radius), (radius, 0), (radius, -radius), (0, -radius)]
                    for k, (dy, dx) in enumerate(points):
                        if image[i + dy, j + dx] >= center:
                            binary_pattern |= (1 << k)
                    lbp_values.append(binary_pattern)
            return np.var(lbp_values) if lbp_values else 0
        lbp_variance = calculate_lbp_variance(center_region)
        binary_trabecular = center_region > np.percentile(center_region, 60)
        cleaned = morphology.remove_small_objects(binary_trabecular, min_size=50)
        labeled_image = measure.label(cleaned)
        num_components = len(np.unique(labeled_image)) - 1
        euler_number = measure.euler_number(cleaned)
        def box_counting_fractal(image):
            sizes, counts = [], []
            for box_size in [2, 4, 8, 16]:
                if box_size < min(image.shape) // 2:
                    count = 0
                    for i in range(0, image.shape[0], box_size):
                        for j in range(0, image.shape[1], box_size):
                            box = image[i:i+box_size, j:j+box_size]
                            if box.size > 0 and np.any(box):
                                count += 1
                    if count > 0:
                        sizes.append(box_size)
                        counts.append(count)
            if len(sizes) > 1:
                coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
                return abs(coeffs[0])
            return 1.5
        fractal_dim = box_counting_fractal(cleaned)
        lbp_risk = max(0, 1 - (lbp_variance / 2000.0))
        connectivity_risk = max(0, 1 - (abs(euler_number) / (num_components + 1)))
        fractal_risk = abs(fractal_dim - 1.85) / 0.5
        trabecular_risk = np.mean([lbp_risk, connectivity_risk, min(1.0, fractal_risk)])
        return trabecular_risk, {'lbp_variance': lbp_variance, 'connectivity_components': num_components, 'euler_number': euler_number, 'fractal_dimension': fractal_dim}

    def analyze_radiolucency(self, image):
        enhanced = self.preprocess_xray(image)
        bone_mask = self.segment_bone(enhanced)
        bone_pixels = enhanced[bone_mask > 0]
        if len(bone_pixels) > 0:
            mean_bone_density = np.mean(bone_pixels)
            bone_density_std = np.std(bone_pixels)
            hist, bins = np.histogram(bone_pixels, bins=50)
            normal_bone_intensity = 180
            radiolucency_score = max(0, 1 - (mean_bone_density / normal_bone_intensity))
            uniformity_score = min(1.0, bone_density_std / 50.0)
            dark_threshold = np.percentile(bone_pixels, 25)
            dark_regions = np.sum(bone_pixels < dark_threshold) / len(bone_pixels)
            overall_radiolucency_risk = np.mean([radiolucency_score, uniformity_score, dark_regions])
        else:
            overall_radiolucency_risk = 0.5
            mean_bone_density, bone_density_std = 0, 0
        return overall_radiolucency_risk, {'mean_density': mean_bone_density, 'density_std': bone_density_std, 'bone_area_ratio': np.sum(bone_mask > 0) / bone_mask.size, 'bone_pixels': bone_pixels if len(bone_pixels) > 0 else np.array([])}

    def detect_compression_fractures(self, image):
        enhanced = self.preprocess_xray(image)
        bone_mask = self.segment_bone(enhanced)
        edges = cv2.Canny(enhanced, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=bone_mask)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vertebral_candidates, compression_indicators = [], []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2500:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 < aspect_ratio < 2.0:
                    vertebral_candidates.append((x, y, w, h, contour))
                    compression_score = self._analyze_vertebral_compression(contour, enhanced[y:y+h, x:x+w])
                    compression_indicators.append(compression_score)
        if compression_indicators:
            max_compression_risk = max(compression_indicators)
            avg_compression_risk = np.mean(compression_indicators)
            overall_risk = max(avg_compression_risk, max_compression_risk * 0.8)
        else:
            overall_risk = 0.0
        return overall_risk, vertebral_candidates, compression_indicators

    def _analyze_vertebral_compression(self, contour, vertebral_roi):
        if vertebral_roi.size == 0:
            return 0.0
        h, w = vertebral_roi.shape
        height_profiles = []
        for x_pos in [w//4, w//2, 3*w//4]:
            if x_pos < w:
                profile = vertebral_roi[:, x_pos]
                non_zero = np.where(profile > np.mean(profile))[0]
                if len(non_zero) > 0:
                    vertebral_height = non_zero[-1] - non_zero[0]
                    height_profiles.append(vertebral_height)
        if len(height_profiles) >= 3:
            anterior_height, middle_height, posterior_height = height_profiles
            anterior_ratio = anterior_height / middle_height if middle_height > 0 else 1
            posterior_ratio = posterior_height / middle_height if middle_height > 0 else 1
            wedging_score = max(0, 1 - min(anterior_ratio, posterior_ratio))
            height_loss_score = max(0, 1 - (middle_height / (h * 0.8)))
            compression_score = np.mean([wedging_score, height_loss_score])
        else:
            compression_score = 0.0
        moments = cv2.moments(contour)
        if moments['m00'] > 0:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour)
            solidity = contour_area / hull_area if hull_area > 0 else 1
            biconcavity_score = max(0, 1 - solidity) * 2
            compression_score = max(compression_score, biconcavity_score * 0.5)
        return min(1.0, compression_score)

    def analyze_endplate_irregularities(self, image):
        enhanced = self.preprocess_xray(image)
        bone_mask = self.segment_bone(enhanced)
        edges = cv2.Canny(enhanced, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=bone_mask)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        irregularity_scores = []
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                smoothness_score = self._calculate_contour_irregularity(contour)
                irregularity_scores.append(smoothness_score)
        if irregularity_scores:
            avg_irregularity = np.mean(irregularity_scores)
            max_irregularity = max(irregularity_scores)
            overall_irregularity_risk = (avg_irregularity + max_irregularity) / 2
        else:
            overall_irregularity_risk = 0.0
        return overall_irregularity_risk, irregularity_scores

    def _calculate_contour_irregularity(self, contour):
        if len(contour) < 10:
            return 0.0
        contour_points = contour.reshape(-1, 2)
        curvatures = []
        for i in range(1, len(contour_points) - 1):
            p1, p2, p3 = contour_points[i-1], contour_points[i], contour_points[i+1]
            v1, v2 = p1 - p2, p3 - p2
            v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                curvatures.append(angle)
        if curvatures:
            curvature_variance = np.var(curvatures)
            irregularity_score = min(1.0, curvature_variance / (np.pi/4)**2)
            return irregularity_score
        return 0.0

    def analyze_bone_geometry_alteration(self, image):
        enhanced = self.preprocess_xray(image)
        bone_mask = self.segment_bone(enhanced)
        contours, _ = cv2.findContours(bone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        geometry_alterations = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                alteration_score = self._analyze_geometric_features(contour)
                geometry_alterations.append(alteration_score)
        if geometry_alterations:
            overall_geometry_risk = np.mean(geometry_alterations)
        else:
            overall_geometry_risk = 0.0
        return overall_geometry_risk, geometry_alterations

    def _analyze_geometric_features(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area == 0 or perimeter == 0:
            return 0.0
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        extent = area / (w * h) if (w * h) > 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        compactness = (perimeter * perimeter) / area if area > 0 else 0
        normal_ranges = {'circularity': (0.3, 0.8), 'aspect_ratio': (0.4, 2.5), 'extent': (0.5, 0.9), 'solidity': (0.8, 1.0), 'compactness': (10, 50)}
        deviations = []
        for prop, (min_val, max_val) in normal_ranges.items():
            value = locals()[prop] if prop in locals() else 0
            if value < min_val:
                deviation = (min_val - value) / min_val
            elif value > max_val:
                deviation = (value - max_val) / max_val
            else:
                deviation = 0.0
            deviations.append(min(1.0, deviation))
        return np.mean(deviations)

    def extract_features(self, image):
        cortical_risk, _, _ = self.analyze_cortical_thinning(image)
        trabecular_risk, _ = self.analyze_trabecular_bone(image)
        radiolucency_risk, _ = self.analyze_radiolucency(image)
        compression_risk, _, _ = self.detect_compression_fractures(image)
        irregularity_risk, _ = self.analyze_endplate_irregularities(image)
        geometry_risk, _ = self.analyze_bone_geometry_alteration(image)
        features = [cortical_risk, trabecular_risk, radiolucency_risk, compression_risk, irregularity_risk, geometry_risk]
        while len(features) < 100:
            features.append(0.0)
        return np.array(features)

    def get_feature_based_prediction(self, features):
        # Simple threshold-based prediction using feature scores
        weighted_risk = np.mean(features[:6])  # Average of the 6 risk factors
        return weighted_risk, self.categorize_overall_risk(weighted_risk)

    def get_model_prediction(self, features):
        # Prediction using the trained model
        features_tensor = torch.tensor(features).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = self.model(features_tensor)
            model_score = torch.sigmoid(output).item()
        return model_score, self.categorize_overall_risk(model_score)

    def combine_predictions(self, feature_score, model_score):
        # Combine feature-based and model-based predictions with equal weighting
        combined_score = (feature_score + model_score) / 2
        return combined_score, self.categorize_overall_risk(combined_score)

    def categorize_overall_risk(self, risk_score):
        if risk_score < 0.25:
            return 'Low Risk'
        elif risk_score < 0.5:
            return 'Moderate Risk'
        elif risk_score < 0.75:
            return 'High Risk'
        else:
            return 'Critical Risk'

    def identify_primary_concerns(self, results):
        factor_risks = {k: results[k]['risk_score'] for k in results if k != 'overall_risk'}
        return [k for k, v in factor_risks.items() if v > self.risk_thresholds.get(k.lower().replace(' ', '_'), 0.5)]

    def comprehensive_osteoporosis_analysis(self, image):
        print("Starting comprehensive osteoporosis analysis...")
        results = {}
        cortical_risk, cortical_measurements, cortical_mask = self.analyze_cortical_thinning(image)
        print("Cortical thinning analysis completed.")
        results['cortical_thinning'] = {'risk_score': cortical_risk, 'measurements': cortical_measurements, 'mask': cortical_mask}
        trabecular_risk, trabecular_metrics = self.analyze_trabecular_bone(image)
        print("Trabecular bone analysis completed.")
        results['trabecular_degradation'] = {'risk_score': trabecular_risk, 'metrics': trabecular_metrics}
        radiolucency_risk, density_metrics = self.analyze_radiolucency(image)
        print("Radiolucency analysis completed.")
        results['radiolucency'] = {'risk_score': radiolucency_risk, 'metrics': density_metrics}
        compression_risk, vertebral_candidates, compression_scores = self.detect_compression_fractures(image)
        print("Compression fractures analysis completed.")
        results['compression_fractures'] = {'risk_score': compression_risk, 'vertebral_candidates': vertebral_candidates, 'compression_scores': compression_scores}
        irregularity_risk, irregularity_scores = self.analyze_endplate_irregularities(image)
        print("Endplate irregularities analysis completed.")
        results['endplate_irregularities'] = {'risk_score': irregularity_risk, 'scores': irregularity_scores}
        geometry_risk, geometry_alterations = self.analyze_bone_geometry_alteration(image)
        print("Geometry alteration analysis completed.")
        results['geometry_alterations'] = {'risk_score': geometry_risk, 'alterations': geometry_alterations}
        
        # Extract features
        features = self.extract_features(image)
        
        # Get individual predictions
        feature_score, feature_category = self.get_feature_based_prediction(features)
        model_score, model_category = self.get_model_prediction(features)
        combined_score, combined_category = self.combine_predictions(feature_score, model_score)
        
        # Store results
        results['overall_risk'] = {
            'feature_based_score': feature_score,
            'feature_based_category': feature_category,
            'model_score': model_score,
            'model_category': model_category,
            'combined_score': combined_score,
            'combined_category': combined_category,
            'primary_concerns': self.identify_primary_concerns(results)
        }
        print("Comprehensive analysis completed.")
        return results

    def visualize_analysis(self, image, results):
        """Visualize the analysis results with a simplified Grad-CAM heatmap for Streamlit."""
        try:
            import torch
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            import cv2
        except ImportError as e:
            raise ImportError(f"Required library missing: {str(e)}. Please install with 'pip install torch numpy matplotlib opencv-python'.")

        h, w = image.shape[:2]
        fig = plt.figure(figsize=(15, 10))

        # Original X-ray
        plt.subplot(2, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original X-ray')
        plt.axis('off')

        # Cortical Thinning with Mask Overlay
        plt.subplot(2, 3, 2)
        plt.imshow(image, cmap='gray')
        cortical_mask = results['cortical_thinning']['mask']
        if cortical_mask is not None and cortical_mask.shape == (h, w):
            plt.imshow(cortical_mask, cmap='Reds', alpha=0.4)
        plt.title(f"Cortical Thinning (Risk: {results['cortical_thinning']['risk_score']:.2f})")
        plt.axis('off')

        # Trabecular Degradation
        plt.subplot(2, 3, 3)
        plt.imshow(image[h//4:3*h//4, w//4:3*w//4], cmap='gray')
        plt.title(f"Trabecular Degradation (Risk: {results['trabecular_degradation']['risk_score']:.2f})")
        plt.axis('off')

        # Radiolucency
        plt.subplot(2, 3, 4)
        plt.imshow(image, cmap='gray')
        plt.title(f"Radiolucency (Risk: {results['radiolucency']['risk_score']:.2f})")
        plt.axis('off')

        # Compression Fractures with Boxes
        plt.subplot(2, 3, 5)
        plt.imshow(image, cmap='gray')
        for x, y, w_box, h_box, _ in results['compression_fractures']['vertebral_candidates']:
            plt.gca().add_patch(Rectangle((x, y), w_box, h_box, edgecolor='red', facecolor='none', lw=2))
        plt.title(f"Compression Fractures (Risk: {results['compression_fractures']['risk_score']:.2f})")
        plt.axis('off')

        # Endplate Irregularities
        plt.subplot(2, 3, 6)
        plt.imshow(image, cmap='gray')
        plt.title(f"Endplate Irregularities (Risk: {results['endplate_irregularities']['risk_score']:.2f})")
        plt.axis('off')

        plt.tight_layout()
        # Return the figure for Streamlit instead of showing it
        return fig

    def _generate_streamlit_gradcam_heatmap(self, image, results):
        """Generate a simplified Grad-CAM heatmap for Streamlit compatibility."""
        try:
            import torch
            import numpy as np
            import matplotlib.pyplot as plt
            import cv2
            from scipy.ndimage import gaussian_filter

            # Set model to evaluation mode
            self.model.eval()
            device = next(self.model.parameters()).device

            # Extract features
            features_np = self.extract_features(image)
            features = torch.from_numpy(features_np).float().to(device)
            features = features.unsqueeze(0)  # Add batch dimension
            features.requires_grad_(True)

            # Forward pass with gradient computation
            with torch.enable_grad():
                output = self.model(features)
                score = torch.sigmoid(output).squeeze()
                score.backward()

                # Get gradients
                gradients = features.grad
                if gradients is None:
                    raise RuntimeError("Gradients are None.")

                # Process gradients
                gradients = gradients.squeeze(0).detach().cpu().numpy()
                importance_weights = np.abs(gradients)
                if np.max(importance_weights) > 0:
                    importance_weights = importance_weights / np.max(importance_weights)

                # Create bone mask (simplified from _create_precise_bone_mask)
                gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                bone_mask = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)
                bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel)

                # Create heatmap
                h, w = image.shape[:2]
                heatmap = np.zeros((h, w), dtype=np.float32)
                bone_regions = bone_mask > 0
                if np.sum(bone_regions) > 0:
                    # Distribute importance based on bone intensity
                    bone_intensities = gray[bone_regions]
                    mean_intensity = np.mean(bone_intensities)
                    y_coords, x_coords = np.where(bone_regions)
                    for y, x in zip(y_coords, x_coords):
                        intensity_factor = gray[y, x] / mean_intensity if mean_intensity > 0 else 1.0
                        heatmap[y, x] = importance_weights[0] * intensity_factor  # Use first weight for simplicity

                # Smooth heatmap
                heatmap = gaussian_filter(heatmap, sigma=2.0)
                heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
                if np.max(heatmap) > 0:
                    heatmap = heatmap / np.max(heatmap)

                # Create figure
                fig = plt.figure(figsize=(8, 6))
                plt.imshow(image, cmap='gray')
                plt.imshow(heatmap, cmap='jet', alpha=0.5)
                plt.title('Grad-CAM Heatmap')
                plt.axis('off')
                plt.colorbar(label='Importance')
                plt.tight_layout()
                return fig

        except Exception as e:
            print(f"Warning: Failed to generate Grad-CAM heatmap: {str(e)}")
            return None

    def _generate_gradcam_heatmap(self, image, results):
        """Generate Grad-CAM heatmap with proper gradient handling."""
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        
        if self.model is None:
            raise ValueError("Model is not initialized. Please train or load the model first.")
        
        # Set model to evaluation mode but enable gradients
        self.model.eval()
        
        # Get model device
        device = next(self.model.parameters()).device
        print(f"Debug: Model device: {device}")
        
        # Preprocess image
        img_processed = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-7)
        
        # Extract features and create proper tensor
        features_np = self.extract_features(image)
        print(f"Debug: extract_features output shape: {features_np.shape}, type: {type(features_np)}")
        
        # Create input tensor with proper gradient requirements
        features = torch.from_numpy(features_np).float().to(device)
        features = features.unsqueeze(0)  # Add batch dimension
        features.requires_grad_(True)  # Enable gradients
        
        print(f"Debug: Input features shape: {features.shape}, requires_grad: {features.requires_grad}")
        
        # Forward pass with gradient computation enabled
        with torch.enable_grad():
            # Clear any existing gradients
            if features.grad is not None:
                features.grad.zero_()
            
            # Forward pass
            output = self.model(features)
            print(f"Debug: Output shape: {output.shape}, requires_grad: {output.requires_grad}")
            
            # Get the prediction (assuming binary classification or regression)
            if output.dim() > 1 and output.size(1) > 1:
                # Multi-class case - use the predicted class
                predicted_class = torch.argmax(output, dim=1)
                score = output[0, predicted_class]
            else:
                # Binary classification or regression case
                score = output.squeeze()
            
            print(f"Debug: Score for backprop: {score}, requires_grad: {score.requires_grad}")
            
            # Backward pass
            score.backward(retain_graph=True)
            
            # Get gradients
            gradients = features.grad
            print(f"Debug: Gradients shape: {gradients.shape if gradients is not None else 'None'}")
            
            if gradients is None:
                raise RuntimeError("Gradients are None. Check if the model has trainable parameters and the computational graph is intact.")
            
            # Process gradients for visualization
            gradients = gradients.squeeze(0)  # Remove batch dimension
            
            # Create importance weights (simple approach for FC layers)
            importance_weights = torch.abs(gradients).detach().cpu().numpy()
            
            # Normalize importance weights
            if np.max(importance_weights) > 0:
                importance_weights = importance_weights / np.max(importance_weights)
            
            # Create spatial heatmap
            heatmap = self._create_spatial_heatmap(importance_weights, image, img_processed)
            
            # Visualize the heatmap
            self._display_gradcam_results(image, heatmap, importance_weights)
    
    def _create_precise_bone_mask(self, image):
        """Create a precise bone mask using advanced segmentation techniques."""
        import cv2
        import numpy as np
        from scipy import ndimage
        from skimage import filters, morphology, measure
        
        # Ensure image is in the right format
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize image
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Method 1: Otsu's thresholding for initial bone segmentation
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive thresholding for local variations
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 21, 2)
        
        # Method 3: Multi-level thresholding for bone intensity
        # Bones typically have higher intensity in X-rays
        bone_threshold = np.percentile(gray, 85)  # Top 15% intensity pixels
        high_intensity_mask = gray > bone_threshold
        
        # Method 4: Gradient-based edge detection for bone boundaries
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Combine different segmentation approaches
        combined_mask = np.zeros_like(gray, dtype=np.uint8)
        
        # Use high intensity regions as primary bone indicator
        combined_mask[high_intensity_mask] = 255
        
        # Refine with Otsu thresholding
        combined_mask = cv2.bitwise_and(combined_mask, otsu_thresh)
        
        # Remove noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small components (noise)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask)
        min_area = 100  # Minimum area for bone components
        
        refined_mask = np.zeros_like(combined_mask)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                refined_mask[labels == i] = 255
        
        # Additional refinement for X-ray specific bone detection
        refined_mask = self._refine_bone_mask_xray(refined_mask, gray)
        
        return refined_mask
    
    def _refine_bone_mask_xray(self, mask, original_image):
        """Additional refinement specifically for X-ray bone detection."""
        import cv2
        import numpy as np
        
        # Create a more conservative bone mask
        refined_mask = mask.copy()
        
        # Method 1: Intensity-based refinement
        # Bones should have consistently high intensity
        mean_intensity = cv2.mean(original_image, mask=mask)[0]
        intensity_threshold = max(mean_intensity * 0.8, np.percentile(original_image, 80))
        
        # Remove regions with low intensity
        refined_mask[original_image < intensity_threshold] = 0
        
        # Method 2: Texture-based refinement
        # Calculate local standard deviation to identify homogeneous bone regions
        kernel = np.ones((5, 5), np.float32) / 25
        mean_filtered = cv2.filter2D(original_image.astype(np.float32), -1, kernel)
        sqr_filtered = cv2.filter2D((original_image.astype(np.float32))**2, -1, kernel)
        local_std = np.sqrt(sqr_filtered - mean_filtered**2)
        
        # Bone regions should have relatively low texture variation
        texture_threshold = np.percentile(local_std, 60)
        high_texture_mask = local_std > texture_threshold
        
        # Remove high texture regions (likely soft tissue or artifacts)
        refined_mask[high_texture_mask] = 0
        
        # Method 3: Shape-based refinement
        # Apply morphological operations to clean up the mask
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel_open)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Method 4: Size-based filtering
        # Remove components that are too small or too large to be bones
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask)
        
        final_mask = np.zeros_like(refined_mask)
        image_area = refined_mask.shape[0] * refined_mask.shape[1]
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Keep components that are reasonable bone sizes
            if 200 < area < image_area * 0.3:  # Between 200 pixels and 30% of image
                final_mask[labels == i] = 255
        
        return final_mask
    
    def _create_spatial_heatmap(self, importance_weights, original_image, processed_image):
        """Create precise spatial heatmap focused only on bone regions."""
        import cv2
        import numpy as np
        from scipy.ndimage import gaussian_filter
        
        h, w = original_image.shape[:2]
        
        # Create precise bone mask
        precise_bone_mask = self._create_precise_bone_mask(original_image)
        
        # Visualize the bone mask for debugging
        self._debug_bone_mask(original_image, precise_bone_mask)
        
        # Create heatmap only in bone regions
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Find bone regions
        bone_regions = precise_bone_mask > 0
        
        if np.sum(bone_regions) > 0:
            # Method 1: Distribute importance based on bone density
            bone_intensities = original_image[bone_regions]
            
            if len(importance_weights) > 1:
                # Create multiple importance zones within bone regions
                self._create_importance_zones(heatmap, bone_regions, importance_weights, 
                                            original_image, precise_bone_mask)
            else:
                # Single importance value
                heatmap[bone_regions] = importance_weights[0] if len(importance_weights) > 0 else 0.5
        
        # Smooth the heatmap within bone regions only
        heatmap = self._smooth_bone_heatmap(heatmap, precise_bone_mask)
        
        # Normalize
        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap
    
    def _create_importance_zones(self, heatmap, bone_regions, importance_weights, original_image, bone_mask):
        """Create different importance zones within bone regions."""
        import cv2
        import numpy as np
        from scipy.ndimage import label, distance_transform_edt
        
        # Find connected bone components
        labeled_bones, num_bones = label(bone_regions)
        
        # Assign different importance weights to different bone regions
        for bone_id in range(1, min(num_bones + 1, len(importance_weights) + 1)):
            bone_component = labeled_bones == bone_id
            
            if np.sum(bone_component) > 0:
                weight_idx = min(bone_id - 1, len(importance_weights) - 1)
                base_importance = importance_weights[weight_idx]
                
                # Create gradient within each bone region based on bone density
                bone_intensities = original_image[bone_component]
                mean_intensity = np.mean(bone_intensities)
                
                # Modulate importance based on local bone density
                y_coords, x_coords = np.where(bone_component)
                for y, x in zip(y_coords, x_coords):
                    local_intensity = original_image[y, x]
                    # Higher intensity = higher importance (denser bone)
                    intensity_factor = local_intensity / mean_intensity if mean_intensity > 0 else 1.0
                    heatmap[y, x] = base_importance * intensity_factor

    def _smooth_bone_heatmap(self, heatmap, bone_mask):
        """Smooth heatmap only within bone regions."""
        import cv2
        import numpy as np
        from scipy.ndimage import gaussian_filter
        
        # Create a smoothed version
        smoothed = gaussian_filter(heatmap, sigma=2.0)
        
        # Apply smoothing only to bone regions
        result = heatmap.copy()
        bone_regions = bone_mask > 0
        result[bone_regions] = smoothed[bone_regions]
        
        return result
    
    def _debug_bone_mask(self, original_image, bone_mask):
        """Debug visualization of bone mask."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Bone mask
        plt.subplot(1, 3, 2)
        plt.imshow(bone_mask, cmap='gray')
        plt.title('Precise Bone Mask')
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(original_image, cmap='gray')
        plt.imshow(bone_mask, cmap='Reds', alpha=0.3)
        plt.title('Bone Mask Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        total_pixels = original_image.shape[0] * original_image.shape[1]
        bone_pixels = np.sum(bone_mask > 0)
        print(f"Bone mask statistics:")
        print(f"  Total pixels: {total_pixels}")
        print(f"  Bone pixels: {bone_pixels}")
        print(f"  Bone percentage: {bone_pixels/total_pixels*100:.2f}%")
    
    def _display_gradcam_results(self, image, heatmap, importance_weights):
        """Display Grad-CAM visualization results with precise bone focus."""
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np
        
        # Create precise bone mask for overlay
        bone_mask = self._create_precise_bone_mask(image)
        
        # Convert heatmap to color (only in bone regions)
        heatmap_masked = heatmap * (bone_mask > 0).astype(np.float32)
        heatmap_uint8 = np.uint8(255 * heatmap_masked)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Make non-bone regions transparent in the heatmap
        colored_heatmap[bone_mask == 0] = [0, 0, 0]
        
        # Superimpose on original image
        if len(image.shape) == 3:
            overlay_image = image.copy()
        else:
            overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Create a more refined overlay
        superimposed_img = overlay_image.copy()
        bone_regions = bone_mask > 0
        if np.sum(bone_regions) > 0:
            # Only overlay where there are bone regions with significant importance
            significant_regions = (heatmap_masked > 0.1) & bone_regions
            superimposed_img[significant_regions] = cv2.addWeighted(
                overlay_image[significant_regions], 0.5, 
                colored_heatmap[significant_regions], 0.5, 0
            )
        
        # Create comprehensive visualization
        plt.figure(figsize=(20, 12))
        
        # Original image
        plt.subplot(2, 4, 1)
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        plt.title('Original X-ray')
        plt.axis('off')
        
        # Bone mask
        plt.subplot(2, 4, 2)
        plt.imshow(bone_mask, cmap='gray')
        plt.title('Detected Bone Regions')
        plt.axis('off')
        
        # Raw heatmap
        plt.subplot(2, 4, 3)
        plt.imshow(heatmap, cmap='jet')
        plt.title('Importance Heatmap')
        plt.axis('off')
        plt.colorbar(label='Importance')
        
        # Masked heatmap (bone regions only)
        plt.subplot(2, 4, 4)
        plt.imshow(heatmap_masked, cmap='jet')
        plt.title('Bone-Focused Heatmap')
        plt.axis('off')
        plt.colorbar(label='Importance')
        
        # Bone overlay
        plt.subplot(2, 4, 5)
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        plt.imshow(bone_mask, cmap='Reds', alpha=0.3)
        plt.title('Bone Mask Overlay')
        plt.axis('off')
        
        # Colored heatmap
        plt.subplot(2, 4, 6)
        plt.imshow(colored_heatmap)
        plt.title('Colored Importance Map')
        plt.axis('off')
        
        # Final superimposed result
        plt.subplot(2, 4, 7)
        plt.imshow(superimposed_img)
        plt.title('Precise Bone Grad-CAM')
        plt.axis('off')
        
        # Feature importance plot
        plt.subplot(2, 4, 8)
        plt.bar(range(len(importance_weights)), importance_weights)
        plt.title('Feature Importance Weights')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        
        plt.tight_layout()
        plt.show()
        
        # Additional statistics
        bone_coverage = np.sum(bone_mask > 0) / (bone_mask.shape[0] * bone_mask.shape[1])
        important_bone_coverage = np.sum(heatmap_masked > 0.1) / np.sum(bone_mask > 0) if np.sum(bone_mask > 0) > 0 else 0
        
        print(f"\nPrecise Bone Analysis:")
        print(f"  Bone region coverage: {bone_coverage*100:.1f}% of image")
        print(f"  Important bone regions: {important_bone_coverage*100:.1f}% of detected bone")
        print(f"  Max importance in bones: {np.max(heatmap_masked):.3f}")
        print(f"  Mean importance in bones: {np.mean(heatmap_masked[bone_mask > 0]):.3f}" if np.sum(bone_mask > 0) > 0 else "  No bone regions detected")
    
    def _create_final_visualization(self, image, results, cortical_mask):
        """Create final annotated visualization."""
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np
        
        h, w = image.shape[:2]
        
        # Create annotated overlay
        if len(image.shape) == 3:
            overlay = image.copy()
        else:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Add cortical thinning mask
        if cortical_mask is not None and cortical_mask.shape == (h, w):
            overlay[cortical_mask > 0] = [255, 0, 0]  # Red for cortical thinning
        
        # Add compression fracture boxes
        for x, y, w_box, h_box, _ in results['compression_fractures']['vertebral_candidates']:
            cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), (0, 255, 255), 2)  # Yellow boxes
        
        # Display final result
        plt.figure(figsize=(12, 8))
        plt.imshow(overlay)
        plt.title('Final Annotated Risk Assessment')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\nOverall Osteoporosis Risk: {results['overall_risk']['combined_category']} (Score: {results['overall_risk']['combined_score']:.2f})")
        print("Primary Concerns:", results['overall_risk']['primary_concerns'])
    
    def evaluate_accuracy(self, test_loader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                print(f"Processing batch {i+1}/{len(test_loader)}")
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                batch_size = images.size(0)
                batch_features = torch.zeros((batch_size, 100)).to(DEVICE)
                for j, img in enumerate(images):
                    features = self.extract_features(img.cpu().numpy())
                    if len(features) == 100:  # Ensure feature length matches
                        batch_features[j] = torch.tensor(features).to(DEVICE)
                    else:
                        batch_features[j] = torch.zeros(100).to(DEVICE)
                outputs = self.model(batch_features)
                predicted = (torch.sigmoid(outputs) > 0.5).float().squeeze()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        if not all_preds or not all_labels:
            print("Warning: No valid predictions or labels. Accuracy evaluation skipped.")
            return 0.0
        accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
        print(f'Accuracy: {accuracy:.2f}%')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1-Score: {f1:.2f}')
        return accuracy
    

    def generate_osteoporosis_report(self, results):
        import google.generativeai as genai

        """Generate a natural language report using Gemini Pro via the SDK."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("API key not found. Please set GEMINI_API_KEY in your .env file or environment variables.")
    
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-1.5-flash')
    
        prompt_text = f"""
        Generate a detailed osteoporosis analysis report based on the following data:
        - Overall Risk Assessment: {results['overall_risk']['combined_category']} (Score: {results['overall_risk']['combined_score']:.2f})
        - Detailed Findings:
          - Cortical Thinning: {results['cortical_thinning']['risk_score']:.2f}
          - Trabecular Degradation: {results['trabecular_degradation']['risk_score']:.2f}
          - Radiolucency: {results['radiolucency']['risk_score']:.2f}
          - Compression Fractures: {results['compression_fractures']['risk_score']:.2f}
          - Endplate Irregularities: {results['endplate_irregularities']['risk_score']:.2f}
          - Geometry Alterations: {results['geometry_alterations']['risk_score']:.2f}
        - Prediction Breakdown:
          - Feature-Based Prediction: {results['overall_risk']['feature_based_category']} (Score: {results['overall_risk']['feature_based_score']:.2f})
          - Model Prediction: {results['overall_risk']['model_category']} (Score: {results['overall_risk']['model_score']:.2f})
          - Combined Prediction: {results['overall_risk']['combined_category']} (Score: {results['overall_risk']['combined_score']:.2f})
        - Primary Concerns: {', '.join(results['overall_risk']['primary_concerns']) if results['overall_risk']['primary_concerns'] else 'None'}
    
        Please provide:
        1. A professional summary.
        2. Key concern explanation.
        3. Suggested next steps.
        """
    
        try:
            response = model.generate_content(prompt_text)
            return response.text
        except Exception as e:
            return f"Error generating report using Gemini SDK: {str(e)}"
    

     


# Example usage
if __name__ == "__main__":
    print(f"Starting execution at 11:44 PM IST, Thursday, July 10, 2025")

    # Load the trained model
    model_path = 'best_osteoporosis_model.pth'  # Ensure this file is in the same directory
    system = OsteoporosisEarlyDetectionSystem(model_path)

    # Define test directories
    healthy_test_dir = r"C:\Users\vishwamohan\Downloads\dataset\healthy_test"
    osteo_test_dir = r"C:\Users\vishwamohan\Downloads\dataset\osteoporosis_test"

    # Check if directories exist, create them if not
    for dir_path in [healthy_test_dir, osteo_test_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}. Please add test images (e.g., .png, .jpg, .jpeg) to this folder.")

    # Prepare test data
    if os.path.exists(healthy_test_dir) and os.path.exists(osteo_test_dir):
        healthy_test_paths = [os.path.join(healthy_test_dir, f) for f in os.listdir(healthy_test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        osteo_test_paths = [os.path.join(osteo_test_dir, f) for f in os.listdir(osteo_test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not healthy_test_paths or not osteo_test_paths:
            print("Warning: No test images found in healthy_test or osteoporosis_test. Please add images.")
        else:
            test_paths = healthy_test_paths + osteo_test_paths
            test_labels = [0.0] * len(healthy_test_paths) + [1.0] * len(osteo_test_paths)

            # Create test dataset and loader
            test_dataset = XrayTestDataset(test_paths, test_labels, transform=system.transform)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

            # Evaluate accuracy
            accuracy = system.evaluate_accuracy(test_loader)
            print(f'Accuracy on Test Set: {accuracy:.2f}%')
    else:
        print("Error: Test directories not found or not fully set up. Please create and populate them.")

    # Optional: Test a single image
    sample_image_path = r"C:\Users\vishwamohan\Downloads\dataset\osteoporosis\n19.jpg"
    print(f"Attempting to load image from: {sample_image_path}")
    image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Failed to load image at {sample_image_path}")
        image = np.zeros(IMG_SIZE, dtype=np.uint8)
    else:
        image = cv2.resize(image, IMG_SIZE)
    print("Image loaded and resized. Starting analysis...")
    results = system.comprehensive_osteoporosis_analysis(image)
    print("\n--- Prediction Summary ---")
    print(f"Feature-Based Score: {results['overall_risk']['feature_based_score']:.2f} ({results['overall_risk']['feature_based_category']})")
    print(f"Model Score: {results['overall_risk']['model_score']:.2f} ({results['overall_risk']['model_category']})")
    print(f"Combined Score: {results['overall_risk']['combined_score']:.2f} ({results['overall_risk']['combined_category']})")
    system.visualize_analysis(image, results)
    
    # Generate and print the report
    report = system.generate_osteoporosis_report(results)
    print("\n--- Osteoporosis Report ---")
    print(report)
