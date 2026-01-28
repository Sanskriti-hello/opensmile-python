import opensmile
import numpy as np
import pandas as pd
import soundfile as sf


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)
audio_path = r"D:\Documents\emotions_v1\Actor_06\03-01-02-02-02-01-06.wav"
signal, sampling_rate = sf.read(audio_path)
df = smile.process_signal(signal, sampling_rate)
pitch_col = [c for c in df.columns if 'F0' in c]

print("\n--- EXTRACTION COMPLETE ---")
print(f"Dataframe shape: {df.shape} (Frames, Features)")
print("\nFirst 5 frames of data:")
print(df.head())
print(f"\nPitch column found: {pitch_col}")