"use client"

import { Upload, X, FileAudio } from "lucide-react"
import { useCallback, useState } from "react"
import { Button } from "@/components/ui/button"

interface FileUploadProps {
  onFileSelect: (file: File | null) => void
  accept?: string
}

export function FileUpload({ onFileSelect, accept = "audio/*" }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)

      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile && droppedFile.type.startsWith("audio/")) {
        setFile(droppedFile)
        onFileSelect(droppedFile)
      }
    },
    [onFileSelect]
  )

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      onFileSelect(selectedFile)
    }
  }

  const removeFile = () => {
    setFile(null)
    onFileSelect(null)
  }

  return (
    <div className="w-full">
      {!file ? (
        <label
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors ${
            isDragging
              ? "border-primary bg-primary/5"
              : "border-border hover:border-primary/50 hover:bg-muted/50"
          }`}
        >
          <Upload className="mb-4 h-10 w-10 text-muted-foreground" />
          <p className="mb-1 text-sm font-medium">Drop your audio file here</p>
          <p className="text-xs text-muted-foreground">or click to browse</p>
          <p className="mt-2 text-xs text-muted-foreground">
            Supports: WAV, MP3, OGG, FLAC
          </p>
          <input
            type="file"
            accept={accept}
            onChange={handleFileInput}
            className="hidden"
          />
        </label>
      ) : (
        <div className="flex items-center justify-between rounded-lg border border-border bg-muted/50 p-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
              <FileAudio className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="text-sm font-medium">{file.name}</p>
              <p className="text-xs text-muted-foreground">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={removeFile}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  )
}
