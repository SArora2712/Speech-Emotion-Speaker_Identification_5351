'use client';

import { useState } from 'react';
import { Upload, Mic, Loader2, Play, Square } from 'lucide-react';

interface AnalysisResult {
  speaker: string;
  emotion: string;
  confidence: number;
  speech_type: string;
  speech_confidence: string;
  sentence: string;
}

export default function AudioAnalyzer() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [fileName, setFileName] = useState('');
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);

  // File Upload
  const analyzeFile = async (file: File) => {
    setLoading(true);
    setError('');
    setFileName(file.name);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Failed to analyze');
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError('Failed to connect to backend. Is the server running?');
    } finally {
      setLoading(false);
    }
  };

  // Microphone Recording (5 seconds)
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];

      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = async () => {
        const audioBlob = new Blob(chunks, { type: 'audio/wav' });
        const file = new File([audioBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        
        setIsRecording(false);
        await analyzeFile(file);
      };

      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);

      // Auto stop after 5 seconds
      setTimeout(() => {
        if (recorder.state === 'recording') recorder.stop();
        stream.getTracks().forEach(track => track.stop());
      }, 5000);

    } catch (err) {
      setError('Microphone access denied or not available');
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-6">
      <div className="text-center mb-10">
        <h2 className="text-4xl font-bold mb-3">Live Demo - EmotiVoice</h2>
        <p className="text-zinc-400">Upload a file or record directly from microphone</p>
      </div>

      {/* Upload & Record Buttons */}
      <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
        <label className="cursor-pointer bg-white text-black px-8 py-4 rounded-2xl font-medium hover:bg-zinc-200 transition flex items-center justify-center gap-3">
          <Upload size={22} />
          Upload .wav File
          <input
            type="file"
            accept="audio/wav"
            onChange={(e) => e.target.files && analyzeFile(e.target.files[0])}
            className="hidden"
          />
        </label>

        <button
          onClick={isRecording ? stopRecording : startRecording}
          className={`px-8 py-4 rounded-2xl font-medium flex items-center justify-center gap-3 transition ${
            isRecording 
              ? 'bg-red-600 hover:bg-red-700' 
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {isRecording ? (
            <>
              <Square size={22} /> Stop Recording (5s)
            </>
          ) : (
            <>
              <Mic size={22} /> Record 5 Seconds
            </>
          )}
        </button>
      </div>

      {error && <p className="text-red-500 text-center mb-6">{error}</p>}

      {/* Loading */}
      {loading && (
        <div className="text-center py-12">
          <Loader2 className="animate-spin mx-auto mb-4 text-blue-500" size={50} />
          <p className="text-lg">Analyzing your audio...</p>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="bg-zinc-900 border border-zinc-700 rounded-3xl p-8 shadow-2xl">
          <div className="flex items-center gap-3 mb-6">
            <Play className="text-green-500" size={28} />
            <h3 className="text-2xl font-semibold">Analysis Result</h3>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <p className="text-zinc-400 text-sm">Speaker</p>
              <p className="text-4xl font-bold mt-1">{result.speaker}</p>
            </div>
            <div>
              <p className="text-zinc-400 text-sm">Emotion</p>
              <p className="text-4xl font-bold mt-1">
                {result.emotion} 
                <span className="text-xl font-normal text-emerald-400 ml-3">
                  ({(result.confidence * 100).toFixed(1)}%)
                </span>
              </p>
            </div>
          </div>

          <div className="mt-6">
            <p className="text-zinc-400 text-sm">Speech Type</p>
            <p className="text-2xl mt-1">
              {result.speech_type} <span className="text-zinc-500">({result.speech_confidence})</span>
            </p>
          </div>

          <div className="mt-8 pt-6 border-t border-zinc-700">
            <p className="text-zinc-400 text-sm mb-3">Final Output</p>
            <p className="text-lg leading-relaxed text-zinc-100">
              {result.sentence}
            </p>
          </div>

          <button
            onClick={() => setResult(null)}
            className="mt-8 w-full py-4 bg-zinc-800 hover:bg-zinc-700 rounded-2xl transition font-medium"
          >
            Analyze Another Audio
          </button>
        </div>
      )}
    </div>
  );
}