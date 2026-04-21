import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import type { LucideIcon } from "lucide-react"

interface ResultCardProps {
  title: string
  value: string
  confidence?: number
  icon: LucideIcon
  color?: string
}

export function ResultCard({
  title,
  value,
  confidence,
  icon: Icon,
  color = "text-primary",
}: ResultCardProps) {
  return (
    <Card className="border-border/60 shadow-sm transition-all duration-200 hover:shadow-md hover:border-border">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
          <Icon className={`h-4 w-4 ${color}`} />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="mb-2 text-2xl font-semibold tracking-tight">{value}</p>
        {confidence !== undefined && (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Confidence</span>
              <span>{confidence}%</span>
            </div>
            <Progress value={confidence} className="h-1.5" />
          </div>
        )}
      </CardContent>
    </Card>
  )
}
