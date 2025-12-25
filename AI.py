import os
import requests
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter

# -------------------------
# Config
# -------------------------
KB_len = -1
GEN_LEN = 500
CKPT_PATH = "persona_binding_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 3
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 1024
LR = 5e-3
NUM_EPOCHS = 1

# -------------------------
# 1. Generic Q&A Data Fetching
# -------------------------
def fetch_hf_data():
    """Extracts questions and answers from a generic Q&A dataset."""
    # Expanded generic Q&A with 100+ pairs covering diverse topics
    generic_qa = """
    what is python? python is a high-level interpreted programming language known for simplicity readability and extensive libraries used in web development data science and automation.
    who created linux? linus torvalds created the linux operating system kernel in 1991 as a free open-source alternative to proprietary unix systems.
    where is silicon valley? silicon valley is located in the san francisco bay area of california and is the global center for technology innovation and venture capital.
    when was the internet invented? the internet was invented in the late 1960s with arpanet which connected research institutions and evolved into the modern internet by the 1980s.
    why do we use databases? we use databases to efficiently store organize retrieve and manage large amounts of structured data with support for transactions and concurrent access.
    how does machine learning work? machine learning uses algorithms and statistical models to analyze patterns in data enabling computers to improve performance on tasks without explicit programming.
    what is artificial intelligence? artificial intelligence is technology that enables machines to simulate human intelligence including learning reasoning problem solving perception and language understanding.
    who invented the telephone? alexander graham bell invented the first practical telephone in 1876 revolutionizing long-distance communication and connecting people across vast distances.
    where is the eiffel tower? the eiffel tower is located in paris france on the champ de mars near the seine river and was built in 1889 for the world's fair.
    when was world war two? world war two lasted from 1939 to 1945 involving most nations of the world forming two opposing military alliances the allies and the axis.
    why is water important? water is essential for all known forms of life serving as a solvent for biochemical reactions regulating temperature and enabling cellular processes in organisms.
    how do computers process information? computers process information using binary code transistors and logical circuits executing instructions through the fetch-decode-execute cycle in the cpu.
    what is the capital of france? the capital of france is paris a major european city known for art fashion gastronomy and culture with landmarks like the louvre and notre-dame.
    who wrote hamlet? william shakespeare wrote the tragedy of hamlet prince of denmark around 1600 exploring themes of revenge madness and moral corruption.
    where do penguins live? penguins live primarily in antarctica and cold southern hemisphere regions including south america south africa australia and new zealand with adaptations for aquatic life.
    when was the first airplane flight? the wright brothers achieved the first powered controlled sustained heavier-than-air flight on december 17 1903 at kitty hawk north carolina.
    why is the sky blue? the sky appears blue due to rayleigh scattering where shorter blue wavelengths of sunlight scatter more than other colors when passing through atmosphere.
    how does photosynthesis work? photosynthesis converts light energy into chemical energy using chlorophyll in plants combining carbon dioxide and water to produce glucose and oxygen.
    what is democracy? democracy is a system of government where power is vested in the people who exercise authority directly or through elected representatives in free elections.
    who discovered penicillin? alexander fleming discovered penicillin antibiotics in 1928 accidentally finding that mold killed bacteria revolutionizing medicine and saving millions of lives.
    what is gravity? gravity is the fundamental force that attracts objects with mass toward each other described by newton's law and einstein's general relativity as spacetime curvature.
    who painted the mona lisa? leonardo da vinci painted the mona lisa during the italian renaissance period between 1503 and 1519 creating one of the most famous artworks in history.
    where is mount everest? mount everest is located in the himalayas on the nepal-tibet border standing at 8849 meters making it the highest mountain peak on earth.
    when was the declaration of independence signed? the declaration of independence was signed on july 4 1776 announcing american colonies' separation from great britain and establishing the united states.
    why do seasons change? seasons change due to earth's axial tilt of 23.5 degrees as it orbits the sun causing different hemispheres to receive varying amounts of sunlight throughout the year.
    how do vaccines work? vaccines work by training the immune system to recognize and fight specific pathogens using weakened or dead microorganisms or their components to stimulate antibody production.
    what is quantum physics? quantum physics studies the behavior of matter and energy at atomic and subatomic scales where particles exhibit wave-particle duality and quantum superposition.
    who founded microsoft? bill gates and paul allen founded microsoft corporation in 1975 initially developing software for personal computers and later dominating the operating system market.
    where is the great wall of china? the great wall of china stretches across northern china spanning approximately 21000 kilometers built over centuries to protect against invasions from nomadic groups.
    when did the moon landing occur? the first moon landing occurred on july 20 1969 when apollo 11 astronauts neil armstrong and buzz aldrin walked on the lunar surface.
    what is dna? dna or deoxyribonucleic acid is the molecule that carries genetic information in living organisms consisting of two strands forming a double helix structure.
    who invented the lightbulb? thomas edison invented the first practical incandescent lightbulb in 1879 though multiple inventors contributed to the development of electric lighting technology.
    where is the amazon rainforest? the amazon rainforest is located in south america primarily in brazil but extends into peru colombia venezuela ecuador bolivia guyana suriname and french guiana.
    when was the printing press invented? johannes gutenberg invented the movable-type printing press around 1440 in germany revolutionizing information distribution and enabling mass production of books.
    why is biodiversity important? biodiversity is important because it maintains ecosystem stability provides resources supports food chains enables adaptation to environmental changes and offers genetic variety.
    how does the heart work? the heart works as a muscular pump with four chambers that circulate blood through the body delivering oxygen and nutrients while removing waste products.
    what is climate change? climate change refers to long-term alterations in temperature precipitation patterns and weather events primarily caused by human activities increasing greenhouse gas emissions.
    who discovered electricity? benjamin franklin and many scientists contributed to understanding electricity with franklin's 1752 kite experiment demonstrating lightning's electrical nature though electricity existed naturally.
    where is the sahara desert? the sahara desert is located in northern africa covering approximately 9 million square kilometers making it the largest hot desert in the world.
    when was the first computer built? the first electronic general-purpose computer eniac was built in 1945 at the university of pennsylvania weighing 30 tons and using vacuum tubes.
    what is evolution? evolution is the process by which species change over time through genetic variation natural selection and adaptation as described by charles darwin's theory.
    who composed the ninth symphony? ludwig van beethoven composed the ninth symphony in 1824 including the famous ode to joy choral finale despite being almost completely deaf.
    where is the statue of liberty? the statue of liberty is located on liberty island in new york harbor gifted by france to the united states in 1886 symbolizing freedom and democracy.
    when did dinosaurs go extinct? dinosaurs went extinct approximately 66 million years ago during the cretaceous-paleogene extinction event likely caused by an asteroid impact and volcanic activity.
    why do we sleep? we sleep to restore energy consolidate memories regulate hormones support immune function repair tissues and process emotions with different sleep stages serving specific purposes.
    how does wifi work? wifi works by transmitting data using radio waves on specific frequencies allowing wireless devices to connect to networks and the internet through routers and access points.
    what is renewable energy? renewable energy comes from naturally replenishing sources like solar wind hydro geothermal and biomass that don't deplete resources and produce minimal environmental impact.
    who wrote don quixote? miguel de cervantes wrote don quixote published in two parts in 1605 and 1615 creating one of the earliest and most influential novels in western literature.
    where is the panama canal? the panama canal is located in panama central america connecting the atlantic and pacific oceans and completed in 1914 after decades of construction.
    when was the television invented? the television was invented in the 1920s with philo farnsworth and john logie baird making key contributions to electronic television technology.
    what is nanotechnology? nanotechnology is the manipulation of matter at the atomic and molecular scale typically between 1 and 100 nanometers enabling new materials and applications.
    who developed the theory of relativity? albert einstein developed the theory of relativity with special relativity in 1905 and general relativity in 1915 revolutionizing physics and our understanding of spacetime.
    where is machu picchu? machu picchu is located in the andes mountains of peru built by the inca empire in the 15th century and rediscovered in 1911 by hiram bingham.
    when was the printing of paper money started? paper money was first printed in china during the tang dynasty around 600 ad though it became widespread during the song dynasty.
    why is exercise important? exercise is important for maintaining cardiovascular health building muscle strength improving mental health regulating weight supporting bone density and reducing disease risk.
    how does blockchain work? blockchain works as a decentralized distributed ledger that records transactions in blocks linked cryptographically ensuring transparency security and immutability without central authority.
    what is the big bang theory? the big bang theory explains the origin of the universe suggesting it began as an extremely hot dense point approximately 13.8 billion years ago and has been expanding ever since.
    who invented the steam engine? james watt significantly improved the steam engine in the 1760s though thomas newcomen built the first practical steam engine in 1712 for pumping water from mines.
    where is the colosseum? the colosseum is located in rome italy built between 70 and 80 ad as an amphitheater for gladiatorial contests and public spectacles holding up to 80000 spectators.
    when was antibiotics discovered? antibiotics were discovered in 1928 when alexander fleming noticed penicillium mold killed bacteria though widespread medical use began in the 1940s during world war two.
    what is cryptocurrency? cryptocurrency is digital or virtual currency using cryptography for security operating on decentralized networks based on blockchain technology like bitcoin and ethereum.
    who painted the sistine chapel? michelangelo painted the sistine chapel ceiling between 1508 and 1512 creating one of the greatest masterpieces of renaissance art including the creation of adam.
    where is angkor wat? angkor wat is located in cambodia built in the 12th century as a hindu temple later converted to buddhism and is the largest religious monument in the world.
    when was the wheel invented? the wheel was invented around 3500 bc in mesopotamia initially used for pottery making and later adapted for transportation revolutionizing trade and travel.
    why is education important? education is important for developing critical thinking skills enabling economic opportunities promoting social mobility fostering informed citizenship and advancing human knowledge and culture.
    how does solar energy work? solar energy works by converting sunlight into electricity using photovoltaic cells or concentrating solar power systems capturing photons and generating electrical current.
    what is artificial neural network? artificial neural networks are computing systems inspired by biological neural networks using interconnected nodes to process information learn patterns and make predictions through training.
    who discovered america? christopher columbus reached the americas in 1492 though indigenous peoples lived there for millennia and viking explorers like leif erikson arrived around 1000 ad.
    where is the louvre museum? the louvre museum is located in paris france originally built as a fortress in the 12th century and is the world's largest art museum housing the mona lisa.
    when was the automobile invented? the automobile was invented in 1885 when karl benz created the first practical gasoline-powered car though earlier steam-powered vehicles existed in the 18th century.
    what is machine vision? machine vision is the technology enabling computers to interpret and understand visual information from the world using cameras sensors and algorithms for automated inspection and analysis.
    who wrote pride and prejudice? jane austen wrote pride and prejudice published in 1813 exploring themes of love class and social expectations in regency-era england.
    where is stonehenge? stonehenge is located in wiltshire england built in stages between 3000 and 2000 bc as a prehistoric monument whose exact purpose remains debated.
    when was paper invented? paper was invented in china around 105 ad by cai lun during the han dynasty revolutionizing written communication and record keeping.
    why is the ocean salty? the ocean is salty because rivers carry dissolved minerals from rocks to the sea where evaporation removes water but leaves salts behind accumulating over millions of years.
    how does gps work? gps works using a network of satellites orbiting earth that transmit signals allowing receivers to calculate precise location through trilateration based on signal timing.
    what is gene editing? gene editing is biotechnology that allows precise modification of dna sequences using tools like crispr-cas9 to add delete or alter genetic material in organisms.
    who invented calculus? isaac newton and gottfried leibniz independently invented calculus in the late 17th century developing mathematical methods for analyzing continuous change and accumulation.
    where is taj mahal? the taj mahal is located in agra india built by mughal emperor shah jahan between 1632 and 1653 as a mausoleum for his wife mumtaz mahal.
    when was the transistor invented? the transistor was invented in 1947 at bell laboratories by john bardeen walter brattain and william shockley revolutionizing electronics and enabling modern computing.
    what is the greenhouse effect? the greenhouse effect is the process where atmospheric gases trap heat from the sun warming earth's surface with excess greenhouse gases causing global warming.
    who painted starry night? vincent van gogh painted starry night in 1889 during his stay at an asylum in saint-remy-de-provence france creating one of the most iconic paintings in history.
    where is petra? petra is located in southern jordan carved into rose-red sandstone cliffs by the nabataean kingdom around 300 bc and rediscovered by the west in 1812.
    when was nuclear energy discovered? nuclear energy was discovered through research in the early 20th century with the first nuclear reactor built by enrico fermi in 1942 and atomic bombs in 1945.
    why do leaves change color? leaves change color in autumn when chlorophyll breaks down revealing underlying pigments like carotenoids and anthocyanins as trees prepare for winter dormancy.
    how does the internet work? the internet works as a global network of interconnected computers using tcp/ip protocols to transmit data in packets through routers switches and physical infrastructure.
    what is dark matter? dark matter is invisible matter that doesn't emit light or energy but exerts gravitational effects comprising approximately 85 percent of the universe's matter content.
    who invented the radio? guglielmo marconi is credited with inventing practical radio communication in the 1890s though nikola tesla and others made significant contributions to radio technology.
    where is chichen itza? chichen itza is located in yucatan mexico built by the maya civilization with the famous pyramid el castillo constructed around 600 ad.
    when was the telescope invented? the telescope was invented in 1608 in the netherlands with hans lippershey receiving the first patent and galileo galilei making astronomical observations in 1609.
    what is synthetic biology? synthetic biology is the design and construction of new biological parts devices and systems or redesigning existing natural biological systems for useful purposes.
    who discovered the electron? j j thomson discovered the electron in 1897 through cathode ray tube experiments identifying it as a subatomic particle with negative charge.
    where is the golden gate bridge? the golden gate bridge is located in san francisco california spanning the golden gate strait connecting san francisco to marin county and completed in 1937.
    when was the camera invented? the camera was invented in various forms with the first permanent photograph taken by joseph nicephore niepce in 1826 and daguerreotype process developed in 1839.
    why is the rainforest called the lungs of the earth? rainforests are called the lungs of the earth because they produce approximately 20 percent of the world's oxygen through photosynthesis and absorb massive amounts of carbon dioxide.
    how does quantum computing work? quantum computing uses quantum bits or qubits that can exist in superposition of states enabling parallel processing of information exponentially faster than classical computers for certain problems.
    """
    
    # Parse the Q&A pairs
    lines = [line.strip() for line in generic_qa.strip().split('\n') if line.strip()]
    questions = []
    answers = []
    
    for line in lines:
        if '?' in line:
            parts = line.split('?', 1)
            if len(parts) == 2:
                questions.append(parts[0].strip().lower())
                answers.append(parts[1].strip().lower())
    
    # Extract question types
    question_types = set()
    for q in questions:
        words = q.split()
        if words:
            first_word = words[0]
            if first_word in ['who', 'what', 'where', 'when', 'why', 'how', 'which', 'whom', 'whose', 'is', 'are', 'do', 'does', 'did', 'can', 'could', 'would', 'should']:
                question_types.add(first_word)
    
    # Combine all text
    all_text = " ".join(questions + answers)
    all_words = all_text.split()
    
    print(f"  Loaded {len(questions)} Q&A pairs, {len(all_words)} words")
    return list(question_types), all_words

def extract_methods(words):
    """Detects action-based Methods using morphological heuristics."""
    candidates = []
    action_suffixes = r".+(ing|ed|ion|al|ment|ance|ive|ize)$"
    for w in words:
        if re.match(action_suffixes, w) and len(w) > 4:
            candidates.append(w)
    m_set = {w for w, c in Counter(candidates).items() if c >= 2}
    m_set.update({"act", "translate", "summarize", "generate", "write", "solve"})
    return m_set

# -------------------------
# 2. Logic Gates
# -------------------------
class InhibitoryRenetworker(nn.Module):
    def __init__(self, gap=0.06):
        super().__init__(); self.gap = gap
    def forward(self, activations):
        lead = torch.max(activations, dim=-1, keepdim=True)[0]
        mask = (lead - activations > 0) & (lead - activations < self.gap)
        out = activations.clone()
        out[mask] -= 100.0  
        return out

class CKYInverter:
    def __init__(self, words, w2i, v_size):
        self.matrix = torch.zeros((v_size, v_size), device=device)
        for i in range(len(words)-1):
            w1, w2 = words[i], words[i+1]
            if w1 in w2i and w2 in w2i: self.matrix[w2i[w1], w2i[w2]] = 1.0
    def get_grounding(self, last_id, v_size):
        mask = torch.full((v_size,), -float('inf'), device=device)
        valid = torch.where(self.matrix[last_id] > 0)[0]
        if len(valid) > 0: mask[valid] = 0.0
        else: mask.fill_(0.0)
        return mask

# -------------------------
# 3. Model & Weighted Dataset with Source Tracking
# -------------------------
class PersonaNeuralNet(nn.Module):
    def __init__(self, v_size, e_dim, h_dim, layers):
        super().__init__()
        self.embedding = nn.Embedding(v_size, e_dim)
        self.rnn = nn.GRU(e_dim, h_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(h_dim, v_size)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class PersonaBindingDataset(Dataset):
    def __init__(self, words, w2i, methods, scenarios, seq_len, word_sources):
        """
        word_sources: list where 0=file_data, 1=hf_qa_data
        """
        self.samples = []
        for i in range(len(words) - seq_len):
            ctx, target = words[i:i+seq_len], words[i+seq_len]
            # 3.5x Weight boost for Method + Question Type (Scenario) co-occurrence
            is_bound = any(t in methods for t in ctx+[target]) and any(t in scenarios for t in ctx+[target])
            weight = 3.5 if is_bound else 1.0
            
            # Track source of this sequence
            source = word_sources[i+seq_len]  # 0=file, 1=hf_qa
            
            self.samples.append((
                torch.tensor([w2i[w] for w in ctx]), 
                torch.tensor(w2i[target]), 
                torch.tensor(weight),
                torch.tensor(source)
            ))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# -------------------------
# 4. Aligned Axis Inhibition
# -------------------------
def apply_aligned_inhibition(x, y, vocab_size, sources):
    """
    Inhibit different axes based on data source:
    - Generic Q&A data (source=1): Inhibit sequence axis (dim 1)
    - File data (source=0): Inhibit batch axis (dim 0)
    Ensures alignment between the two data streams.
    """
    batch_size, seq_len = x.shape
    
    # Create masks for each source type
    hf_mask = (sources == 1)  # Generic Q&A data
    file_mask = (sources == 0)  # File data
    
    # For Generic Q&A: inhibit sequence dimension (horizontal axis)
    # Modify positions along sequence for Q&A samples
    for i in range(seq_len):
        if i < batch_size and hf_mask[i % batch_size]:
            idx_batch = i % batch_size
            idx_seq = i % seq_len
            if idx_seq < seq_len:
                # Inhibit by modulating with target
                x[idx_batch, idx_seq] = (y[idx_batch] + x[idx_batch, idx_seq]) % vocab_size
    
    # For File data: inhibit batch dimension (vertical axis)
    # Modify positions along batch for file samples
    for i in range(batch_size):
        if file_mask[i]:
            idx_seq = i % seq_len
            # Inhibit by inverse modulation with target
            y[i] = (y[i] - x[i, idx_seq]) % vocab_size
    
    return x, y

# -------------------------
# 5. Main Engine
# -------------------------
def generate_text(model, inverter, renetworker, seed, w2i, i2w, seq_len):
    model.eval(); v_size = len(i2w)
    gen_ids = [w2i.get(w, 0) for w in seed.lower().split()]
    print(f"\n>> Q&A-Bound Seed: {seed}")
    for _ in range(GEN_LEN):
        inp = torch.tensor([gen_ids[-seq_len:]], device=device)
        with torch.no_grad():
            logits = model(inp)
            clean = renetworker(logits[0])
            grounding = inverter.get_grounding(gen_ids[-1], v_size)
            probs = F.softmax(clean + grounding, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            word = i2w.get(next_id, i2w.get(0, "unknown"))
            gen_ids.append(next_id); print(word, end=' ', flush=True)

if __name__ == "__main__":
    try: 
        local_raw = open(input("Filename: "), "r", encoding="utf-8").read().lower().split()
    except: 
        local_raw = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text.lower().split()
    
    print("Fetching generic Q&A data...")
    hf_labels, hf_qa_words = fetch_hf_data()
    
    if not hf_qa_words:
        print("WARNING: No Q&A data loaded! Using empty list.")
    
    print(f"  Question types found: {hf_labels}")
    print(f"  Total Q&A words: {len(hf_qa_words)}")
    
    # Track source of each word: 0=file, 1=hf_qa
    file_sources = [0] * len(local_raw)
    hf_sources = [1] * len(hf_qa_words)
    
    all_words = (local_raw + hf_qa_words)[:KB_len]
    word_sources = (file_sources + hf_sources)[:KB_len]
    
    vocab = sorted(list(set(all_words)))
    w2i, i2w = {w: i for i, w in enumerate(vocab)}, {i: w for i, w in enumerate(vocab)}
    
    methods = extract_methods(all_words)
    # Question types from Q&A are treated as Scenarios for tangential binding
    scenarios = set(hf_labels)
    
    inverter = CKYInverter(all_words, w2i, len(vocab))
    renetworker = InhibitoryRenetworker()
    loader = DataLoader(
        PersonaBindingDataset(all_words, w2i, methods, scenarios, SEQ_LEN, word_sources), 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    model = PersonaNeuralNet(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\nTraining with aligned inhibition:")
    print(f"  - File data ({len(local_raw)} words): Batch axis inhibition")
    print(f"  - Generic Q&A data ({len(hf_qa_words)} words): Sequence axis inhibition")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for x, y, w, sources in pbar:
            x, y, w, sources = x.to(device), y.to(device), w.to(device), sources.to(device)
            
            # Apply aligned axis inhibition
            x, y = apply_aligned_inhibition(x, y, len(vocab), sources)
            
            optimizer.zero_grad()
            loss = (F.cross_entropy(model(x), y, reduction='none') * w).mean()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\nModel saved to {CKPT_PATH}")
    
    while True:
        seed = input("\nSeed >> ").strip()
        if not seed: break
        generate_text(model, inverter, renetworker, seed, w2i, i2w, SEQ_LEN)
