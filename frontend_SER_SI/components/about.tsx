import { Badge } from "@/components/ui/badge"

const technologies = [
  "Wav2Vec 2.0",
  "BERT",
  "PyTorch",
  "Transformers",
  "Librosa",
  "Scikit-learn",
  "Python",
  "React",
]

export function About() {
  return (
    <section id="about" className="py-24">
      <div className="container mx-auto max-w-6xl px-4">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="mb-6 text-3xl font-bold tracking-tight md:text-4xl">
            About the Project
          </h2>

          <p className="mb-8 text-pretty text-lg leading-relaxed text-muted-foreground">
            This hybrid emotion detection and speaker identification system leverages
            cutting-edge deep learning architectures. By combining Wav2Vec 2.0 for
            speech analysis with BERT for text understanding, we achieve robust
            multimodal emotion recognition. The speaker identification module uses
            advanced voice embeddings to distinguish between multiple speakers with
            high accuracy.
          </p>

          <div className="mb-12">
            <h3 className="mb-4 text-lg font-semibold">Technologies Used</h3>
            <div className="flex flex-wrap justify-center gap-2">
              {technologies.map((tech) => (
                <Badge key={tech} variant="secondary" className="px-3 py-1 text-sm">
                  {tech}
                </Badge>
              ))}
            </div>
          </div>

          <div className="grid gap-8 text-left md:grid-cols-3">
            <div className="rounded-lg border border-border/50 bg-card/50 p-6">
              <h4 className="mb-2 font-semibold">Speech Processing</h4>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Wav2Vec 2.0 extracts rich acoustic features from audio for accurate
                emotion and speaker analysis.
              </p>
            </div>
            <div className="rounded-lg border border-border/50 bg-card/50 p-6">
              <h4 className="mb-2 font-semibold">Text Understanding</h4>
              <p className="text-sm leading-relaxed text-muted-foreground">
                BERT-based models analyze textual content to capture semantic and
                emotional context.
              </p>
            </div>
            <div className="rounded-lg border border-border/50 bg-card/50 p-6">
              <h4 className="mb-2 font-semibold">Hybrid Fusion</h4>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Multi-modal fusion combines speech and text features for enhanced
                prediction accuracy.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
