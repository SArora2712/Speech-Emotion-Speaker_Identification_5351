import { Mail, Github, Linkedin } from "lucide-react"

export function Contact() {
  return (
    <section id="contact" className="border-t border-border/40 py-16">
      <div className="container mx-auto max-w-6xl px-4">
        <div className="mx-auto max-w-xl text-center">
          <h2 className="mb-4 text-2xl font-bold tracking-tight">
            Get in Touch
          </h2>
          <p className="mb-8 text-muted-foreground">
            Have questions or want to collaborate? Reach out through any of these channels.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-6">
            <a
              href="mailto:contact@emotionai.demo"
              className="flex items-center gap-2 text-muted-foreground transition-colors hover:text-foreground"
            >
              <Mail className="h-5 w-5 text-primary" />
              <span className="text-sm">contact@emotionai.demo</span>
            </a>

            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-muted-foreground transition-colors hover:text-foreground"
            >
              <Github className="h-5 w-5 text-primary" />
              <span className="text-sm">GitHub</span>
            </a>

            <a
              href="https://linkedin.com"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-muted-foreground transition-colors hover:text-foreground"
            >
              <Linkedin className="h-5 w-5 text-primary" />
              <span className="text-sm">LinkedIn</span>
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}
