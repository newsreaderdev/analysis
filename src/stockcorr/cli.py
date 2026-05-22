"""Typer CLI for stockcorr."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(add_completion=False, help="Pairwise relationship analysis for US equities.")
plot_app = typer.Typer(help="Visualization commands.")
pairs_app = typer.Typer(help="Pair inspection commands.")
app.add_typer(plot_app, name="plot")
app.add_typer(pairs_app, name="pairs")


def _resolve(tickers: str) -> list[str]:
    from stockcorr.data.universe import resolve_tickers

    return resolve_tickers(tickers)


def _parse_prefilter(spec: str | None) -> dict | None:
    """Parse 'pearson:0.7' -> {'metric':'pearson','min_abs_value':0.7}."""
    if not spec:
        return None
    name, _, thresh = spec.partition(":")
    if not thresh:
        raise typer.BadParameter(f"--prefilter must be 'metric:threshold', got '{spec}'")
    return {"metric": name.strip(), "min_abs_value": float(thresh)}


@app.command()
def fetch(
    tickers: str = typer.Option(..., "--tickers", help="sp500 | ndx | adr100 | all | A,B,C | @file.txt"),
    start: str = typer.Option(..., "--start", help="YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="YYYY-MM-DD"),
    interval: str = typer.Option("1d", "--interval"),
):
    """Download and cache OHLCV for a ticker spec."""
    from stockcorr.data import YFinanceSource

    tlist = _resolve(tickers)
    typer.echo(f"Fetching {len(tlist)} tickers from {start} to {end} ...")
    src = YFinanceSource()
    panel = src.close_panel(tlist, start, end, interval=interval)
    typer.echo(f"Got panel: {panel.shape[0]} dates × {panel.shape[1]} tickers")


@app.command()
def analyze(
    tickers: str = typer.Option(..., "--tickers"),
    metrics: str = typer.Option("pearson", "--metrics", help="comma list, e.g. pearson,coint,half_life"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    prefilter: str = typer.Option(None, "--prefilter", help="e.g. pearson:0.7"),
    benchmark: str = typer.Option(None, "--benchmark", help="ticker symbol for beta/residual metrics"),
    out: Path = typer.Option(Path("results.parquet"), "--out"),
    n_jobs: int = typer.Option(-1, "--n-jobs"),
    interval: str = typer.Option("1d", "--interval"),
):
    """Run a list of metrics and write a long-format parquet."""
    from stockcorr.data import YFinanceSource
    from stockcorr.pipeline import run_metrics

    tlist = _resolve(tickers)
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    typer.echo(f"Fetching {len(tlist)} tickers ...")
    src = YFinanceSource()
    panel = src.close_panel(tlist, start, end, interval=interval)
    typer.echo(f"Panel: {panel.shape}")

    bench_series = None
    if benchmark:
        bench_panel = src.close_panel([benchmark], start, end, interval=interval)
        if bench_panel.shape[1]:
            bench_series = bench_panel.iloc[:, 0].rename(benchmark)

    pf = _parse_prefilter(prefilter)
    typer.echo(f"Running metrics: {metric_list} (prefilter={pf}) ...")
    results = run_metrics(panel, metric_list, prefilter=pf, benchmark=bench_series, n_jobs=n_jobs)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out)
    typer.echo(f"Wrote {len(results)} rows -> {out}")


@plot_app.command("heatmap")
def plot_heatmap_cmd(
    input: Path = typer.Option(..., "--input"),
    metric: str = typer.Option(..., "--metric"),
    out: Path = typer.Option(Path("heatmap.png"), "--out"),
    cluster: bool = typer.Option(True, "--cluster/--no-cluster"),
):
    """Plot a heatmap from a results parquet."""
    from stockcorr.viz.heatmap import plot_heatmap

    df = pd.read_parquet(input)
    plot_heatmap(df, metric=metric, out=out, cluster=cluster)
    typer.echo(f"Wrote {out}")


@plot_app.command("network")
def plot_network_cmd(
    input: Path = typer.Option(..., "--input"),
    metric: str = typer.Option(..., "--metric"),
    out: Path = typer.Option(Path("network.png"), "--out"),
    top: int = typer.Option(200, "--top"),
):
    """Plot a top-K edge network from results."""
    from stockcorr.viz.network import plot_network

    df = pd.read_parquet(input)
    plot_network(df, metric=metric, out=out, top=top)
    typer.echo(f"Wrote {out}")


@pairs_app.command("top")
def pairs_top_cmd(
    input: Path = typer.Option(..., "--input"),
    metric: str = typer.Option(..., "--metric"),
    n: int = typer.Option(20, "--n"),
    ascending: bool = typer.Option(False, "--ascending/--descending"),
):
    """Show top-N pairs by a metric value."""
    df = pd.read_parquet(input)
    sub = df[df["metric"] == metric].sort_values("value", ascending=ascending).head(n)
    typer.echo(sub.to_string(index=False))


@app.command()
def universe(
    name: str = typer.Argument("all", help="sp500 | ndx | adr100 | all"),
    limit: int = typer.Option(20, "--limit"),
):
    """Inspect a universe."""
    from stockcorr.data.universe import (
        sp500_tickers, ndx_tickers, top_adrs, union_universe,
    )

    loaders = {
        "sp500": sp500_tickers,
        "ndx": ndx_tickers,
        "adr100": top_adrs,
        "all": union_universe,
    }
    if name not in loaders:
        raise typer.BadParameter(f"unknown universe '{name}'")
    df = loaders[name]()
    typer.echo(f"{name}: {len(df)} tickers")
    typer.echo(df.head(limit).to_string(index=False))


if __name__ == "__main__":
    app()
