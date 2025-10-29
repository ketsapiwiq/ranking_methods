import argparse
import os
import urllib.request
import json
from pathlib import Path
from rank_comparia.pipeline import RankingPipeline

def check_cache_exists(cache_dir: str) -> bool:
    """Vérifie si le cache contient des données"""
    cache_path = Path(cache_dir)
    return cache_path.exists() and any(cache_path.iterdir())

def download_data(pipeline: RankingPipeline, force: bool = False):
    """Télécharge les données HF"""
    if not force and check_cache_exists(pipeline.cache_dir):
        print(f"Cache exists at {pipeline.cache_dir}, skipping download")
        return
    print("Downloading data from Hugging Face...")
    # Le téléchargement se fait automatiquement lors de l'init de RankingPipeline

def download_models_metadata(force: bool = False):
    """Télécharge models.json depuis comparia.beta.gouv.fr"""
    data_dir = Path("data")
    models_file = data_dir / "models_data.json"
    
    if models_file.exists() and not force:
        print(f"Models metadata already exists at {models_file}, skipping download")
        return
    
    print("Downloading models metadata from comparia.beta.gouv.fr...")
    data_dir.mkdir(exist_ok=True)
    
    url = "https://comparia.beta.gouv.fr/models.json"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            with open(models_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Models metadata saved to {models_file}")
    except Exception as e:
        print(f"Error downloading models metadata: {e}")
        raise
    
def main():
    parser = argparse.ArgumentParser(description="ComparIA Ranking Pipeline CLI")
    parser.add_argument("--download", action="store_true", help="Download data from Hugging Face")
    parser.add_argument("--download-models", action="store_true",
                       help="Download models metadata from comparia.beta.gouv.fr")
    parser.add_argument("--compute-scores", action="store_true", help="Compute ranking scores")
    parser.add_argument("--compute-frugality", action="store_true", help="Compute frugality scores")
    parser.add_argument("--visualizations", action="store_true", help="Generate visualizations")
    parser.add_argument("--export", action="store_true", help="Export data files")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--category", type=str, help="Process specific category")
    parser.add_argument("--force-download", action="store_true", help="Force download even if cache exists")
    
    args = parser.parse_args()
    
    # Configuration
    cache_dir = "cache"
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir
    
    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise ValueError("HF_TOKEN environment variable must be set")
    
    # Si aucun flag spécifique, exécuter tout
    if not any([args.download, args.compute_scores, args.compute_frugality,
                args.visualizations, args.export]):
        args.all = True
    
    # Initialisation du pipeline
    pipeline = RankingPipeline(
        cache_dir=cache_dir,
        token=token,
        method="ml",
        include_votes=True,
        include_reactions=True,
        bootstrap_samples=1000,
        mean_how="token",
        export_path=Path("output"),
        category=args.category
    )
    
    scores = None
    
    # Cas spécial : si --all, on utilise la méthode run() originale qui fait tout
    if args.all:
        print("Running full pipeline...")
        if args.download_models or args.all:
            download_models_metadata(args.force_download)
        download_data(pipeline, args.force_download)
        scores = pipeline.run()
        return
    
    # Exécution selon les flags individuels
    if args.download_models or args.all:
        download_models_metadata(args.force_download)
    
    if args.download:
        download_data(pipeline, args.force_download)
    
    if args.compute_scores or args.all:
        print("Computing ranking scores...")
        scores = pipeline.compute_scores_only()

    if args.compute_frugality or args.all:
        print("Computing frugality scores...")
        if scores is None:
            scores = pipeline.compute_scores_only()
        scores = pipeline.compute_frugality_only(scores)

    if args.visualizations or args.all:
        print("Generating visualizations...")
        if scores is None:
            raise ValueError("Scores must be computed before generating visualizations. Use --compute-scores first.")
        pipeline.generate_visualizations_only(scores)

    if args.export or args.all:
        print("Exporting data...")
        if scores is None:
            raise ValueError("Scores must be computed before exporting. Use --compute-scores first.")
        pipeline.export_data_only(scores)

if __name__ == "__main__":
    main()