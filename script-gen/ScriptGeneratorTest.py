# Whisper 라이브러리를 불러옵니다
import whisper 

# "audio.mp3" 오디오 파일을 로드. "base" 크기의 Whisper 모델을 메모리에 로드합니다.
model = whisper.load_model("base") 

# 모델의 transcribe() 메소드를 사용하여 "audio.mp3" 파일을 음성 인식하여 텍스트로 변환합니다.
# 이 메소드는 전체 파일을 읽고 30초 길이의 윈도우를 이동시키며 오디오를 처리합니다. 
# 각 윈도우에서 자동 회귀 시퀀스-투-시퀀스 예측을 수행합니다.
result = model.transcribe("test.mp3")
print(result["text"])

# 오디오를 30초 길이에 맞게 패딩하거나 자릅니다
audio = whisper.load_audio("testsong.mp3") 
audio = whisper.pad_or_trim(audio)

# 오디오의 로그 멜 스펙트로그램을 생성하고 모델이 있는 같은 디바이스로 이동합니다
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect_language() 메소드를 사용하여 말해진 언어를 감지합니다
_, probs = model.detect_language(mel)
