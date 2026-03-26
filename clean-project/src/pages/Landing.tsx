import { Link } from "react-router-dom";
import { ArrowRight, Shield, BarChart3, Mic, FileText, Activity, Layers, FileUp } from "lucide-react";
import { Button } from "@/components/ui/button";

const features = [
  {
    icon: Mic,
    title: "Call & Chat Analysis",
    desc: "Upload recordings or chat logs and get instant AI-powered transcription and quality scoring.",
  },
  {
    icon: BarChart3,
    title: "Quality Scoring",
    desc: "Six-dimensional scoring across empathy, compliance, resolution, proficiency, and more.",
  },
  {
    icon: Shield,
    title: "Bias Detection",
    desc: "Automatic fairness analysis ensures agents treat all customers equally and consistently.",
  },
  {
    icon: FileText,
    title: "Summary Generation",
    desc: "Instant AI-generated summaries of calls and chats, capturing key points and action items.",
  },
  {
    icon: Layers,
    title: "Batch Transcribing",
    desc: "Upload and transcribe multiple files simultaneously — process entire call batches in one go.",
  },
  {
    icon: FileUp,
    title: "Policy-Based Scoring",
    desc: "Upload your own policy documents and generate quality scores based on your custom compliance rules.",
  },
];

export default function Landing() {
  return (
    <div className="min-h-screen bg-background overflow-hidden">
      {/* Nav */}
      <nav className="relative z-10 flex items-center justify-between max-w-[1200px] mx-auto px-6 py-6">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-blue-500/20 flex items-center justify-center glow-primary">
            <Activity className="h-5 w-5 text-blue-500" />
          </div>
          <span className="text-xl font-heading font-bold">
            Aura <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-cyan-500">Q</span>
          </span>
        </div>
        <div className="hidden md:flex items-center gap-8 text-sm text-muted-foreground">
          <a href="#features" className="hover:text-foreground transition-colors">Features</a>
          <a href="#preview" className="hover:text-foreground transition-colors">Scores</a>
          <a href="#how" className="hover:text-foreground transition-colors">How it works</a>
        </div>
        <Link to="/dashboard">
          <Button className="bg-blue-600 text-white hover:bg-blue-700 rounded-xl px-6">
            Get Started
          </Button>
        </Link>
      </nav>

      {/* Hero */}
      <section className="relative max-w-[1200px] mx-auto px-6 pt-16 pb-32 text-center bg-hero-gradient">
        {/* Decorative orbs - Changed from primary/chart-4 to blue/cyan */}
        <div className="absolute top-10 left-1/4 w-72 h-72 rounded-full bg-blue-500/5 blur-3xl animate-float" />
        <div className="absolute top-20 right-1/4 w-60 h-60 rounded-full bg-cyan-500/5 blur-3xl animate-float" style={{ animationDelay: "2s" }} />

        <div className="relative z-10">
          {/* Logo orb */}
          <div className="flex flex-col items-center justify-center mb-6">
            <div className="flex items-center gap-6 mb-3">
              <div className="relative h-28 w-28 flex-shrink-0 flex items-center justify-center">
                <div className="absolute inset-0 rounded-full border border-blue-500/15" style={{ transform: 'scale(1.5)', boxShadow: '0 0 30px rgba(59, 130, 246, 0.1)' }} />
                <div className="absolute inset-0 rounded-full border border-blue-500/25" style={{ transform: 'scale(1.2)' }} />
                <div className="relative h-20 w-20 rounded-full flex items-center justify-center overflow-hidden" style={{ background: 'radial-gradient(circle at 35% 30%, #60a5fa, #2563eb 50%, #1e40af 100%)', boxShadow: '0 0 40px rgba(37, 99, 235, 0.5), 0 0 80px rgba(37, 99, 235, 0.2), inset 0 -8px 20px #1e3a8a' }}>
                  <div className="absolute top-2 left-4 h-3 w-3 rounded-full bg-white/40 blur-sm" />
                  <svg viewBox="0 0 60 30" className="w-12 h-6 relative z-10" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round">
                    <path d="M5 20 Q10 8 15 15 Q20 22 25 12 Q30 2 35 15 Q40 28 45 12 Q50 2 55 10" opacity="0.9" />
                    <path d="M5 24 Q12 18 20 22 Q28 26 35 20 Q42 14 50 18 Q55 20 58 17" opacity="0.4" strokeWidth="1.5" />
                  </svg>
                </div>
              </div>
              <div className="flex flex-col">
                <h2 className="text-5xl md:text-6xl font-heading font-black tracking-tight text-foreground">
                  Aura<span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-cyan-500">_Q</span>
                </h2>
                <div className="flex items-center gap-2 mt-1">
                  <span className="h-1.5 w-1.5 rounded-full bg-blue-600" />
                  <p className="text-xs tracking-[0.3em] uppercase text-muted-foreground">
                     AI Call Quality Auditor
                  </p>
                </div>
              </div>
            </div>
          </div>

          <h1 className="text-5xl md:text-7xl font-heading font-black leading-tight max-w-4xl mx-auto">
            Read the{" "}
            {/* Swapped text-gradient for a blue gradient */}
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-cyan-500">Aura</span>
            <br />
            of every call.
          </h1>

          <p className="text-lg text-muted-foreground max-w-2xl mx-auto mt-6 leading-relaxed">
            AuraQ listens to your customer calls and instantly scores
            Empathy, Compliance ,Resolution, Proficiency and Bias reduction — giving managers real
            intelligence, not just transcripts.
          </p>

          <div className="flex items-center justify-center gap-4 mt-10">
            <Link to="/dashboard">
              {/* Button: from-blue-600 to-cyan-500 */}
              <Button size="lg" className="rounded-xl px-8 py-6 text-base font-semibold bg-gradient-to-r from-blue-600 to-cyan-600 hover:opacity-90 text-white shadow-lg shadow-blue-500/20">
                Launch AuraQ <ArrowRight className="h-5 w-5 ml-2" />
              </Button>
            </Link>
            <a href="#features">
              <Button size="lg" variant="outline" className="rounded-xl px-8 py-6 text-base border-border/50 text-muted-foreground hover:text-foreground">
                See Features
              </Button>
            </a>
          </div>
        </div>
      </section>

      {/* Product Preview */}
      <section id="preview" className="relative max-w-[1100px] mx-auto px-6 -mt-16">
        <div className="border border-blue-500/20 rounded-2xl p-1 bg-card/30 backdrop-blur-xl">
          <div className="rounded-xl bg-card p-6 md:p-8 space-y-4">
            <div className="flex items-center gap-3 mb-2">
              <div className="h-3 w-3 rounded-full bg-destructive/60" />
              <div className="h-3 w-3 rounded-full bg-yellow-500/60" />
              <div className="h-3 w-3 rounded-full bg-green-500/60" />
              <span className="text-xs text-muted-foreground ml-2 font-mono">AuraQ Dashboard</span>
            </div>
            <div className="grid grid-cols-3 gap-3">
              {["Empathy", "Compliance", "Resolution"].map((label, i) => (
                <div key={label} className="glass-card rounded-xl p-4 border border-white/5">
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</span>
                  <div className="flex items-baseline gap-1 mt-2">
                    <span className="text-2xl font-heading font-bold text-blue-500">{[8, 9, 8][i]}</span>
                    <span className="text-sm text-muted-foreground">/10</span>
                  </div>
                  <div className="h-1.5 w-full bg-muted rounded-full mt-3 overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-blue-500 to-cyan-400" 
                      style={{ width: `${[80, 90, 80][i]}%` }} 
                    />
                  </div>
                </div>
              ))}
            </div>
            <div className="glass-card rounded-xl p-4 flex items-center gap-3 border border-white/5">
              <div className="h-8 w-8 rounded-lg bg-blue-500/15 flex items-center justify-center">
                <Mic className="h-4 w-4 text-blue-500" />
              </div>
              <div className="flex-1">
                <div className="h-2 bg-muted rounded-full w-3/4" />
                <div className="h-2 bg-muted/50 rounded-full w-1/2 mt-2" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="max-w-[1100px] mx-auto px-6 py-24">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-heading font-bold">
            Everything you need to{" "}
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-cyan-500">audit quality</span>
          </h2>
          <p className="text-muted-foreground mt-4 max-w-lg mx-auto">
            From real-time transcription to fairness analysis — one platform for complete quality assurance.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((f) => (
            <div key={f.title} className="glass-card hover:border-blue-500/30 transition-all rounded-2xl p-7 border border-white/5">
              <div className="h-12 w-12 rounded-xl bg-blue-500/10 flex items-center justify-center mb-5">
                <f.icon className="h-6 w-6 text-blue-600" />
              </div>
              <h3 className="text-lg font-heading font-bold mb-2">{f.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section id="how" className="max-w-[1100px] mx-auto px-6 pb-24">
        <div className="rounded-2xl p-10 md:p-16 text-center bg-blue-500/5 border border-blue-500/10">
          <h2 className="text-3xl md:text-4xl font-heading font-bold mb-4">
            Ready to transform your QA?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-lg mx-auto">
            Start auditing calls and chats with AI precision. No setup required.
          </p>
          <Link to="/dashboard">
            <Button size="lg" className="rounded-xl px-10 py-6 text-base font-semibold bg-gradient-to-r from-blue-600 to-cyan-600 hover:opacity-90 text-white shadow-lg shadow-blue-500/20">
              Launch Dashboard <ArrowRight className="h-5 w-5 ml-2" />
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/30 py-8 text-center text-sm text-muted-foreground">
        <p>© 2026 AuraQ. AI-Powered Customer Quality Auditing.</p>
      </footer>
    </div>
  );
}