import multiprocessing as mp
from paper_benchmark import run_seed_task

if __name__ == "__main__":
    seeds = [42, 100, 2026, 777, 12345]
    tasks = [(s, "LHS") for s in seeds]
    print(f"Launching {len(tasks)} LHS tasks in parallel...")
    with mp.Pool(processes=5) as pool:
        pool.map(run_seed_task, tasks)
    print("LHS Re-run complete!")
