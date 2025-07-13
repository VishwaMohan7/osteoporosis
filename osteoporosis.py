import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import morphology, measure
import google.generativeai as genai
from scipy.ndimage import gaussian_filter

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

# OsteoporosisEarlyDetectionSystem for inference
class OsteoporosisEarlyDetectionSystem:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model file is available.")
        self.model = FeatureClassifier(input_size=100).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
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
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        while len(image.shape) > 2 and image.shape[-1] <= 1:
            image = image.squeeze(-1)
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        elif len(image.shape) == 3 and image.shape[-1] in [1, 3]:
            image = image.squeeze(axis=-1) if image.shape[-1] == 1 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) > 2:
            image = image[0] if image.shape[0] == 1 else image
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
        weighted_risk = np.mean(features[:6])
        return weighted_risk, self.categorize_overall_risk(weighted_risk)

    def get_model_prediction(self, features):
        features_tensor = torch.tensor(features).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = self.model(features_tensor)
            model_score = torch.sigmoid(output).item()
        return model_score, self.categorize_overall_risk(model_score)

    def combine_predictions(self, feature_score, model_score):
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
        results = {}
        cortical_risk, cortical_measurements, cortical_mask = self.analyze_cortical_thinning(image)
        results['cortical_thinning'] = {'risk_score': cortical_risk, 'measurements': cortical_measurements, 'mask': cortical_mask}
        trabecular_risk, trabecular_metrics = self.analyze_trabecular_bone(image)
        results['trabecular_degradation'] = {'risk_score': trabecular_risk, 'metrics': trabecular_metrics}
        radiolucency_risk, density_metrics = self.analyze_radiolucency(image)
        results['radiolucency'] = {'risk_score': radiolucency_risk, 'metrics': density_metrics}
        compression_risk, vertebral_candidates, compression_scores = self.detect_compression_fractures(image)
        results['compression_fractures'] = {'risk_score': compression_risk, 'vertebral_candidates': vertebral_candidates, 'compression_scores': compression_scores}
        irregularity_risk, irregularity_scores = self.analyze_endplate_irregularities(image)
        results['endplate_irregularities'] = {'risk_score': irregularity_risk, 'scores': irregularity_scores}
        geometry_risk, geometry_alterations = self.analyze_bone_geometry_alteration(image)
        results['geometry_alterations'] = {'risk_score': geometry_risk, 'alterations': geometry_alterations}
        
        features = self.extract_features(image)
        feature_score, feature_category = self.get_feature_based_prediction(features)
        model_score, model_category = self.get_model_prediction(features)
        combined_score, combined_category = self.combine_predictions(feature_score, model_score)
        
        results['overall_risk'] = {
            'feature_based_score': feature_score,
            'feature_based_category': feature_category,
            'model_score': model_score,
            'model_category': model_category,
            'combined_score': combined_score,
            'combined_category': combined_category,
            'primary_concerns': self.identify_primary_concerns(results)
        }
        return results

    def visualize_analysis(self, image, results):
        h, w = image.shape[:2]
        fig = plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original X-ray')
        plt.axis('off')
        plt.subplot(2, 3, 2)
        plt.imshow(image, cmap='gray')
        cortical_mask = results['cortical_thinning']['mask']
        if cortical_mask is not None and cortical_mask.shape == (h, w):
            plt.imshow(cortical_mask, cmap='Reds', alpha=0.4)
        plt.title(f"Cortical Thinning (Risk: {results['cortical_thinning']['risk_score']:.2f})")
        plt.axis('off')
        plt.subplot(2, 3, 3)
        plt.imshow(image[h//4:3*h//4, w//4:3*w//4], cmap='gray')
        plt.title(f"Trabecular Degradation (Risk: {results['trabecular_degradation']['risk_score']:.2f})")
        plt.axis('off')
        plt.subplot(2, 3, 4)
        plt.imshow(image, cmap='gray')
        plt.title(f"Radiolucency (Risk: {results['radiolucency']['risk_score']:.2f})")
        plt.axis('off')
        plt.subplot(2, 3, 5)
        plt.imshow(image, cmap='gray')
        for x, y, w_box, h_box, _ in results['compression_fractures']['vertebral_candidates']:
            plt.gca().add_patch(Rectangle((x, y), w_box, h_box, edgecolor='red', facecolor='none', lw=2))
        plt.title(f"Compression Fractures (Risk: {results['compression_fractures']['risk_score']:.2f})")
        plt.axis('off')
        plt.subplot(2, 3, 6)
        plt.imshow(image, cmap='gray')
        plt.title(f"Endplate Irregularities (Risk: {results['endplate_irregularities']['risk_score']:.2f})")
        plt.axis('off')
        plt.tight_layout()
        return fig

    def _generate_streamlit_gradcam_heatmap(self, image, results):
        self.model.eval()
        device = next(self.model.parameters()).device
        features_np = self.extract_features(image)
        features = torch.from_numpy(features_np).float().to(device)
        features = features.unsqueeze(0)
        features.requires_grad_(True)
        with torch.enable_grad():
            output = self.model(features)
            score = torch.sigmoid(output).squeeze()
            score.backward()
            gradients = features.grad
            if gradients is None:
                return None
            gradients = gradients.squeeze(0).detach().cpu().numpy()
            importance_weights = np.abs(gradients)
            if np.max(importance_weights) > 0:
                importance_weights = importance_weights / np.max(importance_weights)
            gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            bone_mask = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)
            bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel)
            h, w = image.shape[:2]
            heatmap = np.zeros((h, w), dtype=np.float32)
            bone_regions = bone_mask > 0
            if np.sum(bone_regions) > 0:
                bone_intensities = gray[bone_regions]
                mean_intensity = np.mean(bone_intensities)
                y_coords, x_coords = np.where(bone_regions)
                for y, x in zip(y_coords, x_coords):
                    intensity_factor = gray[y, x] / mean_intensity if mean_intensity > 0 else 1.0
                    heatmap[y, x] = importance_weights[0] * intensity_factor
            heatmap = gaussian_filter(heatmap, sigma=2.0)
            heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(image, cmap='gray')
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            plt.title('Grad-CAM Heatmap')
            plt.axis('off')
            plt.colorbar(label='Importance')
            plt.tight_layout()
            return fig

    def generate_osteoporosis_report(self, results):
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            return "Error: GEMINI_API_KEY not set in environment variables."
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
            return f"Error generating report: {str(e)}. Basic Report:\n" \
                   f"Overall Risk: {results['overall_risk']['combined_category']} (Score: {results['overall_risk']['combined_score']:.2f})\n" \
                   f"Primary Concerns: {', '.join(results['overall_risk']['primary_concerns']) if results['overall_risk']['primary_concerns'] else 'None'}"
