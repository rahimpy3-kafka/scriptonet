import pandas as pd
import numpy as np

def create_mock_corpus(num_samples=100):
    # The 5 mechanism classes from ScriptoNet architecture
    mechanisms = [
        "Scriptotherapy", 
        "Poetry Therapy", 
        "Music Therapy", 
        "Narrative Reconstruction", 
        "Emotional Catharsis"
    ]
    
    sample_texts = [
        "She returned to the page believing that only by writing could she confront the injury she had caused.",
        "The melody curved around the edges of me, and the song said what words could not.",
        "Its verse moved like breath, uneven yet true, carrying a rhythm that steadied me.",
        "I remembered the day in fragments at first, then reordered them with compassion.",
        "When I finally spoke the whole story, the tears came—not from weakness, but release.",
        "He walked down the street looking at the buildings, thinking about what to buy for dinner." # Non-therapeutic example
    ]
    
    data = []
    for i in range(num_samples):
        text_idx = i % len(sample_texts)
        text = sample_texts[text_idx]
        
        # Non-therapeutic examples shouldn't have active mechanism tags or intensity scores
        if "dinner" in text or "street" in text:
            is_therapeutic = 0
            mechanism_class = -1 # Padding class index for non-therapeutic events
            intensity = 0.0
        else:
            is_therapeutic = 1
            mechanism_class = text_idx % len(mechanisms) # Map to a 0-4 class index
            intensity = float(np.random.randint(1, 6)) # 1 to 5 scale
            
        data.append({
            "passage_id": f"MTC_{i:04d}",
            "text": text,
            "is_therapeutic": is_therapeutic,     # Binary detection target
            "mechanism_label": mechanism_class,   # 5-class target
            "intensity_score": intensity          # Regression target
        })
        
    df = pd.DataFrame(data)
    df.to_csv("mcewan_therapeutic_corpus.csv", index=False)
    print("Successfully generated 'mcewan_therapeutic_corpus.csv' for testing!")

if __name__ == "__main__":
    create_mock_corpus(num_samples=100)
