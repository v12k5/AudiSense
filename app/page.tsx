import RecorderPanel from "@/components/recorder-panel"

export default function Page() {
  return (
    <main className="min-h-dvh flex items-center justify-center p-6">
      <div className="w-full max-w-3xl">
        <section className="rounded-xl border bg-card text-card-foreground p-6 md:p-8 shadow-sm">
          <header className="mb-6">
            <h1 className="text-2xl md:text-3xl font-semibold text-balance">Real‑Time Noise Reduction</h1>
            <p className="mt-2 text-muted-foreground text-pretty">
              Record audio, then refine it with an in‑browser Wiener filter that reduces background noise. No backend
              required.
            </p>
          </header>
          <RecorderPanel />
        </section>
      </div>
    </main>
  )
}
