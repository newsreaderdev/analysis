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
    benchmark: str = typer.Option(
        None, "--benchmark",
        help="benchmark ticker(s), comma-separated; first is used for beta, "
             "all jointly for residual_corr (e.g. SPY,XLK,XLF)",
    ),
    regime: str = typer.Option(
        None, "--regime",
        help="regime spec 'TICKER:threshold', e.g. '^VIX:20' labels days high/low "
             "by the ticker's close level (required for regime_corr)",
    ),
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

    bench: pd.Series | pd.DataFrame | None = None
    if benchmark:
        bench_tickers = [t.strip() for t in benchmark.split(",") if t.strip()]
        bench_panel = src.close_panel(bench_tickers, start, end, interval=interval)
        if bench_panel.shape[1] == 1:
            bench = bench_panel.iloc[:, 0].rename(bench_tickers[0])
        elif bench_panel.shape[1] > 1:
            bench = bench_panel

    regime_series = None
    if regime:
        r_ticker, _, r_thresh = regime.partition(":")
        if not r_thresh:
            raise typer.BadParameter(f"--regime must be 'TICKER:threshold', got '{regime}'")
        regime_panel = src.close_panel([r_ticker.strip()], start, end, interval=interval)
        if regime_panel.shape[1]:
            level = regime_panel.iloc[:, 0]
            regime_series = level.gt(float(r_thresh)).map({True: "high", False: "low"})

    pf = _parse_prefilter(prefilter)
    typer.echo(f"Running metrics: {metric_list} (prefilter={pf}) ...")
    results = run_metrics(
        panel, metric_list, prefilter=pf, benchmark=bench, regime=regime_series, n_jobs=n_jobs
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out)
    typer.echo(f"Wrote {len(results)} rows -> {out}")


@app.command()
def screen(
    tickers: str = typer.Option(None, "--tickers", help="sp500 | ndx | adr100 | all | A,B,C | @file.txt"),
    start: str = typer.Option(None, "--start"),
    end: str = typer.Option(None, "--end"),
    input_csv: Path = typer.Option(
        None, "--input-csv",
        help="offline mode: wide CSV (first column = date, one column per ticker) "
             "used instead of fetching from the data source",
    ),
    prefilter_threshold: float = typer.Option(0.65, "--prefilter-threshold", help="|Pearson| floor"),
    max_p: float = typer.Option(0.05, "--max-p", help="Engle-Granger raw p-value gate"),
    use_fdr: bool = typer.Option(False, "--use-fdr", help="gate on FDR-5% instead of raw p (strict)"),
    half_life_min: float = typer.Option(5.0, "--half-life-min"),
    half_life_max: float = typer.Option(60.0, "--half-life-max"),
    max_hurst: float = typer.Option(0.5, "--max-hurst"),
    min_roll_corr: float = typer.Option(0.6, "--min-roll-corr"),
    max_roll_corr_std: float = typer.Option(0.20, "--max-roll-corr-std"),
    top: int = typer.Option(20, "--top", help="finalists to print"),
    out: Path = typer.Option(Path("recommended_pairs.csv"), "--out"),
    n_jobs: int = typer.Option(-1, "--n-jobs"),
    interval: str = typer.Option("1d", "--interval"),
):
    """Run the full pairs-trading screening funnel and write a candidate list.

    Funnel: Pearson prefilter -> Engle-Granger (+FDR) -> split-half stability
    -> half-life window -> Hurst -> rolling-correlation stability.
    """
    from stockcorr.screen import screen_pairs

    if input_csv is not None:
        panel = pd.read_csv(input_csv, index_col=0, parse_dates=True).sort_index()
        typer.echo(f"Loaded local panel: {panel.shape[0]} dates x {panel.shape[1]} tickers")
    else:
        if not (tickers and start and end):
            raise typer.BadParameter("either --input-csv or all of --tickers/--start/--end are required")
        from stockcorr.data import YFinanceSource

        tlist = _resolve(tickers)
        typer.echo(f"Fetching {len(tlist)} tickers ...")
        panel = YFinanceSource().close_panel(tlist, start, end, interval=interval)
        typer.echo(f"Panel: {panel.shape}")

    sectors = None
    try:
        from stockcorr.data.universe import union_universe

        meta = union_universe()
        sectors = meta.set_index("ticker")["sector"]
    except Exception:
        pass

    result = screen_pairs(
        panel,
        prefilter_threshold=prefilter_threshold,
        max_p_value=max_p,
        use_fdr=use_fdr,
        half_life_range=(half_life_min, half_life_max),
        max_hurst=max_hurst,
        min_roll_corr=min_roll_corr,
        max_roll_corr_std=max_roll_corr_std,
        sectors=sectors,
        n_jobs=n_jobs,
    )

    typer.echo("")
    typer.echo(result.summary())
    typer.echo("")
    if result.finalists.empty:
        typer.echo("No pairs survived the funnel. Try a lower --prefilter-threshold, "
                   "a wider half-life window, or check the sample length (>= 2y recommended).")
    else:
        show_cols = [c for c in ["ticker_a", "ticker_b", "sector_a", "p_value", "p_value_fdr",
                                 "hedge_ratio", "half_life_days", "z_score", "hurst",
                                 "roll_corr_mean"] if c in result.finalists.columns]
        typer.echo(f"Top {min(top, len(result.finalists))} of {len(result.finalists)} finalists:")
        typer.echo(result.finalists.head(top)[show_cols].round(4).to_string(index=False))

    out.parent.mkdir(parents=True, exist_ok=True)
    result.finalists.to_csv(out, index=False)
    full_out = out.with_name(out.stem + "_full_table.csv")
    result.table.to_csv(full_out, index=False)
    typer.echo(f"\nFinalists -> {out}")
    typer.echo(f"Full candidate table -> {full_out}")


@app.command()
def backtest(
    tickers: str = typer.Option(None, "--tickers"),
    start: str = typer.Option(None, "--start"),
    end: str = typer.Option(None, "--end"),
    input_csv: Path = typer.Option(
        None, "--input-csv",
        help="offline mode: wide CSV (first column = date, one column per ticker)",
    ),
    pair: list[str] = typer.Option(
        None, "--pair", help="pair as 'A,B'; repeat the flag for several pairs"),
    pairs_csv: Path = typer.Option(
        None, "--pairs-csv",
        help="CSV with ticker_a/ticker_b columns (e.g. output of `stockcorr screen`)"),
    formation: int = typer.Option(252, "--formation", help="trailing estimation window (days)"),
    entry_z: float = typer.Option(2.0, "--entry-z"),
    exit_z: float = typer.Option(0.5, "--exit-z"),
    stop_z: float = typer.Option(3.5, "--stop-z"),
    max_holding: int = typer.Option(None, "--max-holding",
                                    help="time stop in days (default: 2x formation half-life)"),
    cost_bps: float = typer.Option(5.0, "--cost-bps", help="per-side cost on gross notional"),
    out: Path = typer.Option(Path("backtest_trades.csv"), "--out"),
    plot: Path = typer.Option(None, "--plot", help="write equity-curve PNG here"),
    interval: str = typer.Option("1d", "--interval"),
):
    """Walk-forward backtest of one or more pairs (entry/exit/stop/time rules).

    Typical flow: `stockcorr screen ... --out pairs.csv` then
    `stockcorr backtest --pairs-csv pairs.csv ...` on a LATER date range
    (selecting and trading on the same window is look-ahead bias).
    """
    from stockcorr.backtest import backtest_pairs, plot_backtest

    if input_csv is not None:
        panel = pd.read_csv(input_csv, index_col=0, parse_dates=True).sort_index()
        typer.echo(f"Loaded local panel: {panel.shape[0]} dates x {panel.shape[1]} tickers")
    else:
        if not (tickers and start and end):
            raise typer.BadParameter("either --input-csv or all of --tickers/--start/--end are required")
        from stockcorr.data import YFinanceSource

        tlist = _resolve(tickers)
        panel = YFinanceSource().close_panel(tlist, start, end, interval=interval)
        typer.echo(f"Panel: {panel.shape}")

    pair_list: list[tuple[str, str]] = []
    if pairs_csv is not None:
        pdf = pd.read_csv(pairs_csv)
        pair_list += list(zip(pdf["ticker_a"], pdf["ticker_b"]))
    for p in pair or []:
        a, _, b = p.partition(",")
        if not b:
            raise typer.BadParameter(f"--pair must be 'A,B', got '{p}'")
        pair_list.append((a.strip().upper(), b.strip().upper()))
    if not pair_list:
        raise typer.BadParameter("no pairs given: use --pair A,B and/or --pairs-csv file.csv")
    pair_list = [(a, b) for a, b in pair_list if a in panel.columns and b in panel.columns]
    typer.echo(f"Backtesting {len(pair_list)} pair(s) ...")

    result = backtest_pairs(
        panel, pair_list,
        formation_window=formation, entry_z=entry_z, exit_z=exit_z,
        stop_z=stop_z, max_holding_days=max_holding, cost_bps=cost_bps,
    )
    if not result.stats:
        typer.echo("No backtest produced (not enough data after the formation window?)")
        raise typer.Exit(1)

    typer.echo("")
    typer.echo("PORTFOLIO " + "=" * 50)
    typer.echo(result.summary())
    per_pair = result.stats.get("per_pair", {})
    if len(per_pair) > 1:
        typer.echo("")
        typer.echo("PER PAIR " + "=" * 51)
        rows = [{
            "pair": k, "trades": v["n_trades"],
            "win": f"{v['win_rate']:.0%}" if v["n_trades"] else "-",
            "total": f"{v['total_return']:+.2%}", "sharpe": f"{v['sharpe']:.2f}",
            "maxDD": f"{v['max_drawdown']:.1%}",
        } for k, v in per_pair.items()]
        typer.echo(pd.DataFrame(rows).to_string(index=False))

    out.parent.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(out, index=False)
    typer.echo(f"\nTrades -> {out}")
    if plot is not None:
        plot_backtest(result, out=plot)
        typer.echo(f"Equity curve -> {plot}")


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
