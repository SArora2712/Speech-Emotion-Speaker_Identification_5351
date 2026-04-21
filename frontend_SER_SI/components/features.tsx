import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Brain, Mic, MessageSquare, Layers } from "lucide-react"

const features = [
  {
    icon: Mic,
    title: "Speech Emotion Detection",
    description:
      "Analyze audio inputs to detect emotions like happiness, sadness, anger, and more using Wav2Vec 2.0.",
  },
  {
    icon: MessageSquare,
    title: "Text Emotion Analysis",
    description:
      "Process text inputs to identify emotional states using BERT-based transformer models.",
  },
  {
    icon: Brain,
    title: "Speaker Identification",
    description:
      "Identify and distinguish between different speakers based on unique voice characteristics.",
  },
  {
    icon: Layers,
    title: "Hybrid AI Architecture",
    description:
      "Combines multiple deep learning models for more accurate and robust emotion detection.",
  },
]

export function Features() {
  return (
    <section id="features" className="border-y border-border/40 bg-muted/20 py-24">
      <div className="container mx-auto max-w-6xl px-4">
        <div className="mx-auto mb-16 max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight md:text-4xl">
            Core Capabilities
          </h2>
          <p className="text-pretty text-muted-foreground">
            Our system combines state-of-the-art AI models to deliver comprehensive
            emotion and speaker analysis.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {features.map((feature) => (
            <Card
              key={feature.title}
              className="border-border/50 bg-card/50 transition-all hover:border-primary/30 hover:shadow-md"
            >
              <CardHeader>
                <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                  <feature.icon className="h-6 w-6 text-primary" />
                </div>
                <CardTitle className="text-lg">{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-sm leading-relaxed">
                  {feature.description}
                </CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
