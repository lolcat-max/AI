import torch
import numpy as np
from collections import Counter, defaultdict
import math
import os
import xml.etree.ElementTree as ET
from datetime import datetime

# =====================================================================
# CONFIGURATION
# =====================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

N_GRAM_ORDER = 2

# Precision configuration
USE_FLOAT64 = True  # Set to True for float64, False for float32
ENABLE_TF32 = True  # Set to True to enable TF32 for faster computation on Ampere+ GPUs

# Set precision
if USE_FLOAT64:
    torch_dtype = torch.float64
    print("🔬 Using float64 (double precision) for high numerical accuracy")
else:
    torch_dtype = torch.float32
    print("⚡ Using float32 precision")

# Configure TF32 (only affects float32 on Ampere+ GPUs)
if ENABLE_TF32 and not USE_FLOAT64 and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("🚀 TF32 tensor cores enabled for accelerated computation")
elif ENABLE_TF32 and USE_FLOAT64:
    print("ℹ️  TF32 has no effect with float64 precision")
elif not ENABLE_TF32:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    print("🔒 TF32 disabled - using full float32 precision")


# =====================================================================
# N42.42 RADIATION DETECTOR DATA PARSER
# =====================================================================

class N42RadiationParser:
    """
    Parser for ANSI/IEEE N42.42 XML radiation detector data format.
    Supports both spectrum data and dose rate measurements.
    """
    def __init__(self, n42_file=None):
        self.n42_file = n42_file
        self.measurements = []
        self.dose_rates = []
        self.spectra = []
        self.metadata = {}
        self.entropy_source = None
        print("☢️  N42.42 Radiation Data Parser initialized")
        
    def parse_n42_file(self, filepath):
        """Parse N42.42 XML file and extract radiation data"""
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Extract namespace if present
            ns = {'n42': 'http://physics.nist.gov/N42/2011/N42'}
            if not root.tag.startswith('{'):
                ns = {}
            
            # Extract instrument information
            inst_info = root.find('.//n42:RadInstrumentInformation', ns) if ns else root.find('.//RadInstrumentInformation')
            if inst_info is not None:
                manufacturer = inst_info.find('.//n42:RadInstrumentManufacturerName', ns) if ns else inst_info.find('.//RadInstrumentManufacturerName')
                model = inst_info.find('.//n42:RadInstrumentModelName', ns) if ns else inst_info.find('.//RadInstrumentModelName')
                if manufacturer is not None:
                    self.metadata['manufacturer'] = manufacturer.text
                if model is not None:
                    self.metadata['model'] = model.text
                    print(f"📡 Instrument: {manufacturer.text if manufacturer is not None else 'Unknown'} {model.text if model is not None else 'Unknown'}")
            
            # Extract detector information
            detector_info = root.find('.//n42:RadDetectorInformation', ns) if ns else root.find('.//RadDetectorInformation')
            if detector_info is not None:
                det_desc = detector_info.find('.//n42:RadDetectorDescription', ns) if ns else detector_info.find('.//RadDetectorDescription')
                if det_desc is not None:
                    self.metadata['detector'] = det_desc.text
                    print(f"🔬 Detector: {det_desc.text}")
            
            # Extract measurement data
            measurements = root.findall('.//n42:RadMeasurement', ns) if ns else root.findall('.//RadMeasurement')
            
            for measurement in measurements:
                measurement_data = {}
                
                # Extract timestamp
                start_time = measurement.find('.//n42:StartDateTime', ns) if ns else measurement.find('.//StartDateTime')
                if start_time is not None:
                    measurement_data['timestamp'] = start_time.text
                
                # Extract real time
                real_time = measurement.find('.//n42:RealTimeDuration', ns) if ns else measurement.find('.//RealTimeDuration')
                if real_time is not None:
                    measurement_data['real_time'] = float(real_time.text.replace('PT', '').replace('S', ''))
                
                # Extract dose rate
                dose_rate = measurement.find('.//n42:DoseRate/n42:DoseRateValue', ns) if ns else measurement.find('.//DoseRate/DoseRateValue')
                if dose_rate is not None:
                    dose_value = float(dose_rate.text)
                    measurement_data['dose_rate'] = dose_value
                    self.dose_rates.append(dose_value)
                
                # Extract geographic coordinates
                lat = measurement.find('.//n42:LatitudeValue', ns) if ns else measurement.find('.//LatitudeValue')
                lon = measurement.find('.//n42:LongitudeValue', ns) if ns else measurement.find('.//LongitudeValue')
                elev = measurement.find('.//n42:ElevationValue', ns) if ns else measurement.find('.//ElevationValue')
                
                if lat is not None and lon is not None:
                    measurement_data['location'] = {
                        'latitude': float(lat.text),
                        'longitude': float(lon.text),
                        'elevation': float(elev.text) if elev is not None else None
                    }
                
                # Extract spectrum counts (if available)
                spectrum = measurement.find('.//n42:Spectrum', ns) if ns else measurement.find('.//Spectrum')
                if spectrum is not None:
                    channel_data = spectrum.find('.//n42:ChannelData', ns) if ns else spectrum.find('.//ChannelData')
                    if channel_data is not None:
                        counts = [int(x) for x in channel_data.text.split()]
                        measurement_data['counts'] = counts
                        measurement_data['total_counts'] = sum(counts)
                        self.spectra.append(counts)
                
                if measurement_data:
                    self.measurements.append(measurement_data)
            
            print(f"✅ Parsed {len(self.measurements)} measurements from N42.42 file")
            if self.dose_rates:
                print(f"📊 Dose rates: min={min(self.dose_rates):.2f}, max={max(self.dose_rates):.2f}, avg={np.mean(self.dose_rates):.2f}")
            
            # Generate entropy from radiation data
            if self.measurements:
                self._generate_entropy()
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error parsing N42.42 file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_entropy(self):
        """Generate high-quality entropy from radiation measurements"""
        entropy_values = []
        
        # Use dose rates as primary entropy source
        if self.dose_rates:
            # Convert dose rates to integer entropy by multiplying and taking fractional parts
            for dr in self.dose_rates:
                # Extract fine-grained variation by using high precision multiplication
                entropy_values.append(int((dr * 1000000) % 1000000))
        
        # Use spectrum counts if available
        if self.spectra:
            for spectrum in self.spectra:
                entropy_values.extend(spectrum)
        
        # Use timestamps for additional entropy
        for measurement in self.measurements:
            if 'timestamp' in measurement:
                # Use microsecond-level time variations
                try:
                    dt = datetime.fromisoformat(measurement['timestamp'].replace('Z', '+00:00'))
                    entropy_values.append(int(dt.timestamp() * 1000000) % 1000000)
                except:
                    pass
        
        if entropy_values:
            self.entropy_source = np.array(entropy_values, dtype=np.float64)
            print(f"🎲 Generated entropy pool from {len(entropy_values)} radiation measurements")
            print(f"🔐 Entropy range: {min(entropy_values)} to {max(entropy_values)}")
    
    def get_quantum_seed(self):
        """Get a truly random seed from radiation decay events"""
        if self.entropy_source is not None and len(self.entropy_source) > 0:
            # Use current time + entropy to select seed
            idx = int(datetime.now().timestamp() * 1000000) % len(self.entropy_source)
            seed = int(self.entropy_source[idx]) % (2**32)
            return seed
        return None


# =====================================================================
# SINGLE-EQUATION ISOMORPHIC FEATURE EXTRACTOR
# =====================================================================

class SchrodingerQuantumFeatures:
    """
    Single-equation isomorphic feature extractor with radiation data integration.
    """
    def __init__(self, hbar=1.0, radiation_parser=None):
        self.hbar = hbar
        self.device = device
        self.dtype = torch_dtype
        self.radiation_parser = radiation_parser
        print(f"🧮 Single-equation isomorphic feature extractor initialized on {self.device} with {self.dtype}")

    def extract_quantum_features(self, segment, word_freq, total_words):
        """
        Compute all features with optional radiation data influence:
            F = σ(-(x - x̄)/w) * (|x - x̄| + 1/(f/N + ε))
        """
        eps = 1e-10 if self.dtype == torch.float64 else 1e-6
        w = 1.0
        
        # Apply radiation entropy if available
        if self.radiation_parser and self.radiation_parser.entropy_source is not None:
            seed = self.radiation_parser.get_quantum_seed()
            if seed:
                torch.manual_seed(seed)
                np.random.seed(seed % (2**32))
        
        x = torch.tensor([len(wd) for wd in segment],
                         dtype=self.dtype, device=self.device)
        f = torch.tensor([word_freq.get(wd, 1.0) for wd in segment],
                         dtype=self.dtype, device=self.device)
        N = float(total_words)

        # Fixed computation with proper error handling
        try:
            x_mean = x.mean()
            F = torch.sigmoid(-(x - x_mean) / w) * (torch.abs(x - x_mean) + 1.0 / (f / N + eps))
        except Exception as e:
            print(f"Warning: Feature computation error, using fallback: {e}")
            F = torch.ones_like(x) / len(x)

        Z = torch.sum(F) + eps
        F_norm = F / Z
        avg_energy = torch.mean(F_norm).item()
        energy_variance = torch.var(F_norm).item()
        avg_probability = torch.mean(F_norm).item()
        coherence = 1.0 / (1.0 + energy_variance)
        uncertainty_product = torch.std(x).item() * torch.std(F_norm).item()

        return {
            'avg_energy': avg_energy,
            'energy_variance': energy_variance,
            'avg_probability': avg_probability,
            'coherence': coherence,
            'uncertainty_product': uncertainty_product
        }


# =====================================================================
# N-GRAM MODEL BUILDER
# =====================================================================

def build_ngram_model(tokens, n=N_GRAM_ORDER):
    model = defaultdict(list)
    for i in range(len(tokens) - n):
        key = tuple(tokens[i:i + n])
        model[key].append(tokens[i + n])
    return model


# =====================================================================
# TEXT GENERATOR
# =====================================================================

class SimpleGenerator:
    """
    Text generator using n-gram transitions with radiation-enhanced randomness.
    """
    def __init__(self, tokens, model, feature_extractor):
        self.tokens = tokens
        self.model = model
        self.keys = list(model.keys())
        self.feature_extractor = feature_extractor
        self.word_freq = Counter(tokens)
        self.total_words = len(tokens)
        self.dtype = torch_dtype

    def generate(self, seed, length=100):
        # Use radiation seed if available
        if self.feature_extractor.radiation_parser:
            rad_seed = self.feature_extractor.radiation_parser.get_quantum_seed()
            if rad_seed:
                np.random.seed(rad_seed)
        
        if seed not in self.model:
            seed = self.keys[np.random.randint(0, len(self.keys))]
        output = list(seed)

        for _ in range(length):
            candidates = self.model.get(seed, [])
            if not candidates:
                seed = self.keys[_ % len(self.keys)]
                candidates = self.model.get(seed, [])
                if not candidates:
                    continue

            # Use the feature extractor to get coherence weighting
            segment = list(seed)
            coherence_scores = []
            for cand in candidates:
                seg = segment + [cand]
                q = self.feature_extractor.extract_quantum_features(
                    seg, self.word_freq, self.total_words
                )
                coherence_scores.append(q['coherence'])

            probs = torch.tensor(coherence_scores, dtype=self.dtype, device=device)
            probs = torch.softmax(probs, dim=0).cpu().numpy()

            next_word = np.random.choice(candidates, p=probs)
            output.append(next_word)
            seed = tuple(output[-N_GRAM_ORDER:])
        return " ".join(output)


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("\n=== Context-Aware Text Generator with Radiation Data Integration ===")
    print(f"Precision: {torch_dtype}, TF32: {ENABLE_TF32 and not USE_FLOAT64}\n")

    # Optional: Load N42.42 radiation data
    radiation_parser = N42RadiationParser()

    if os.path.exists("n42.xml"):
        if radiation_parser.parse_n42_file("n42.xml"):
            print("☢️  Radiation data loaded - using quantum entropy for generation")
        else:
            print("Failed to parse N42 file, proceeding without radiation data")
            radiation_parser = None

    # Load text corpus
    filename = input("Enter text file: ").strip()
    if not os.path.exists(filename):
        print("File not found.")
        return

    text = open(filename, 'r', encoding='utf-8').read().lower()
    tokens = text.split()
    print(f"Loaded {len(tokens):,} tokens.")

    # Build model
    print("Building n-gram model...")
    model = build_ngram_model(tokens)
    print(f"N-gram model size: {len(model):,} keys.")

    # Initialize feature extractor and generator
    extractor = SchrodingerQuantumFeatures(radiation_parser=radiation_parser)
    generator = SimpleGenerator(tokens, model, extractor)
    
    while True:
        seed_input = input("\nEnter start words (or 'quit' to exit): ").lower().strip()
        if seed_input == 'quit':
            break
            
        seed_input = seed_input.split()[:N_GRAM_ORDER]
        while len(seed_input) < N_GRAM_ORDER:
            seed_input.append(tokens[len(seed_input) % len(tokens)])
        seed = tuple(seed_input)

        print("\n--- Generated Text ---\n")
        output = generator.generate(seed, length=1000)
        print(output)
        print("\n--- End ---")

if __name__ == "__main__":
    main()
