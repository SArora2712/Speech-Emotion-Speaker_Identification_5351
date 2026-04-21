'use client';

import { useState } from 'react';
import { Upload, Mic, Loader2, Play, Square, ArrowLeft, Volume2 } from 'lucide-react';
import Link from 'next/link';

interface AnalysisResult {
  speaker: string;
  emotion: string;
  confidence: number;
  speech_type: string;
  speech_confidence: string;
  sentence: string;
}

export default function DemoPage() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [fileName, setFileName] = useState('');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const analyzeFile = async (file: File) => {
    setLoading(true);
    setError('');
    setFileName(file.name);
    setUploadedFile(file);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Analysis failed');
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError('Backend server is not running. Please start uvicorn on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];

      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = async () => {

        const blob = new Blob(chunks, { type: 'audio/wav' });
        const file = new File([blob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });

        setIsRecording(false);
        await analyzeFile(file);
      };

      recorder.start();
      setIsRecording(true);

      setTimeout(() => {
        if (recorder.state === "recording") recorder.stop();
        stream.getTracks().forEach(track => track.stop());
      }, 5000);
    } catch {
      setError("Microphone access denied or unavailable");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-black to-zinc-950 text-white pb-20">
      <div className="max-w-5xl mx-auto px-6 pt-10">
        {/* Header */}
        <Link href="/" className="inline-flex items-center gap-2 text-zinc-400 hover:text-white mb-10 transition">
          ← Back to Home
        </Link>

        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold tracking-tight mb-4">Analyze Audio</h1>
          <p className="text-xl text-zinc-400 max-w-2xl mx-auto">
            Real-time Emotion, Speaker Identification & Speech Context Analysis
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-zinc-900/80 border border-zinc-700 rounded-3xl p-10 shadow-2xl backdrop-blur-xl">
          {/* Input Area */}
          <div className="mb-12">
            <h3 className="text-2xl font-semibold text-center mb-8">Choose Audio Input</h3>

            <div className="grid md:grid-cols-2 gap-6 max-w-2xl mx-auto">
              {/* Upload Button */}
              <label className="group cursor-pointer">
                <div className="h-64 flex flex-col items-center justify-center border-2 border-dashed border-zinc-700 hover:border-zinc-500 rounded-2xl transition-all hover:bg-zinc-950/50">
                  <Upload className="h-12 w-12 text-zinc-400 group-hover:text-white mb-4" />
                  <p className="font-medium text-lg">Upload .wav File</p>
                  <p className="text-sm text-zinc-500 mt-1">Click or drag file here</p>
                  <input
                    type="file"
                    accept="audio/wav"
                    className="hidden"
                    onChange={(e) => e.target.files && analyzeFile(e.target.files[0])}
                  />
                </div>
              </label>

              {/* Record Button */}
              <button
                onClick={startRecording}
                disabled={isRecording}
                className={`h-64 flex flex-col items-center justify-center border-2 border-dashed rounded-2xl transition-all
                  ${isRecording
                    ? 'border-red-500 bg-red-950/30'
                    : 'border-zinc-700 hover:border-blue-500 hover:bg-zinc-950/50'}`}
              >
                {isRecording ? (
                  <Square className="h-12 w-12 text-red-500 mb-4 animate-pulse" />
                ) : (
                  <Mic className="h-12 w-12 text-blue-500 mb-4" />
                )}
                <p className="font-medium text-lg">
                  {isRecording ? "Recording..." : "Record from Microphone"}
                </p>
                <p className="text-sm text-zinc-500 mt-1">5 seconds recording</p>
              </button>
            </div>
          </div>

          {error && <p className="text-red-500 text-center mb-8 font-medium">{error}</p>}

          {/* File Info */}
          {fileName && !result && (
            <div className="text-center mb-6 text-zinc-400 flex items-center justify-center gap-2">
              <Volume2 size={18} />
              Selected: <span className="text-white font-medium">{fileName}</span>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="text-center py-20">
              <Loader2 className="h-16 w-16 animate-spin mx-auto text-blue-500 mb-6" />
              <p className="text-xl">Analyzing with AI Models...</p>
              <p className="text-zinc-500 mt-2">This may take a few seconds</p>
            </div>
          )}

          {/* Results Section */}
          {result && (
            <div className="space-y-10">
              <div className="text-center">
                <div className="inline-flex items-center gap-2 bg-green-500/10 text-green-400 px-4 py-1.5 rounded-full text-sm font-medium">
                  <Play size={16} /> Analysis Complete
                </div>
              </div>

              <div className="grid md:grid-cols-3 gap-6">
                {/* Emotion Card */}
                <div className="bg-zinc-950 border border-zinc-800 rounded-2xl p-8 text-center">
                  <p className="text-zinc-400 text-sm mb-3">EMOTION</p>
                  <p className="text-5xl font-bold mb-2">{result.emotion}</p>
                  <div className="h-2 bg-zinc-800 rounded-full mt-6 overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-emerald-400 to-cyan-400" style={{ width: `${result.confidence * 100}%` }}></div>
                  </div>
                  <p className="text-emerald-400 mt-2 font-medium">{(result.confidence * 100).toFixed(1)}% Confidence</p>
                </div>

                {/* Speaker Card */}
                <div className="bg-zinc-950 border border-zinc-800 rounded-2xl p-8 text-center">
                  <p className="text-zinc-400 text-sm mb-3">SPEAKER</p>
                  <p className="text-5xl font-bold mb-6">{result.speaker}</p>
                  <div className="text-sm text-zinc-500">Identified Successfully</div>
                </div>

                {/* Speech Type Card */}
                <div className="bg-zinc-950 border border-zinc-800 rounded-2xl p-8 flex flex-col justify-center items-center text-center h-full">
                  <p className="text-zinc-400 text-sm mb-3">SPEECH TYPE</p>
                  <p className="text-5xl font-bold mb-2 capitalize">
                    {result.speech_type}
                  </p>
                  <p className="text-zinc-400">
                    ({result.speech_confidence})
                  </p>
                </div>
              </div>

              {/* Final Output */}
              <div className="bg-zinc-950 border border-zinc-700 rounded-2xl p-8">
                <p className="text-zinc-400 text-sm mb-4">FINAL OUTPUT</p>
                <p className="text-lg leading-relaxed text-zinc-100">
                  {result.sentence}
                </p>
              </div>

              <button
                onClick={() => {
                  setResult(null);
                  setFileName('');
                  setUploadedFile(null);
                }}
                className="w-full py-4 bg-white text-black rounded-2xl font-semibold hover:bg-zinc-200 transition"
              >
                Analyze Another Audio
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}