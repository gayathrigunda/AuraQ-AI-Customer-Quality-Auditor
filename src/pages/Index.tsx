import { useState } from "react";
import { DashboardHeader } from "@/components/DashboardHeader";
import { StatsOverview } from "@/components/StatsOverview";
import { UploadPanel } from "@/components/UploadPanel";
import { ScoreRadarChart } from "@/components/ScoreRadarChart";
import { ScoreCards } from "@/components/ScoreCards";
import { TranscriptViewer } from "@/components/TranscriptViewer";
import { SummaryCard } from "@/components/SummaryCard";
import { AuditHistory } from "@/components/AuditHistory";
import { ScoreTrendChart } from "@/components/ScoreTrendChart";
import { mockAudits, type AuditRecord } from "@/lib/mock-data";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";

const Index = () => {
  const [selected, setSelected] = useState<AuditRecord>(mockAudits[0]);

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader />

      <main className="max-w-[1440px] mx-auto px-4 md:px-6 py-6 space-y-6">
        <StatsOverview />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left column */}
          <div className="space-y-6">
            <UploadPanel />
            <AuditHistory audits={mockAudits} onSelect={setSelected} selectedId={selected.id} />
          </div>

          {/* Center column */}
          <div className="lg:col-span-2 space-y-6">
            {/* Selected audit header */}
            <div className="glass-card rounded-xl p-5 flex items-center justify-between animate-slide-up">
              <div>
                <h2 className="text-xl font-heading font-bold">{selected.title}</h2>
                <p className="text-sm text-muted-foreground mt-1">
                  {selected.agent} • {selected.date} • {selected.duration}
                </p>
              </div>
              <Button variant="outline" size="sm" className="border-border/50 text-muted-foreground hover:text-foreground">
                <Download className="h-4 w-4 mr-2" /> Report
              </Button>
            </div>

            <SummaryCard summary={selected.summary} overallScore={selected.overallScore} />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <ScoreRadarChart scores={selected.scores} />
              <ScoreCards scores={selected.scores} />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <TranscriptViewer transcript={selected.transcript} />
              <ScoreTrendChart />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
