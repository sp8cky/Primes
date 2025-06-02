
def format_timing(times: List[float]) -> str:
    return f"⏱ Best: {min(times)*1000:.2f}ms | Avg: {mean(times)*1000:.2f}ms | Worst: {max(times)*1000:.2f}ms"



def test_protocoll(numbers: List[int], timings: Optional[Dict[str, List[Dict]]] = None):
    for n in numbers:
        print(f"\n\033[1mTeste n = {n}\033[0m")

        # Miller-Rabin
        from src.primality.tests import miller_rabin_test
        result, details = miller_rabin_test(n, k=3)
        print(f"Miller-Rabin: {'✅ Prim' if result else '❌ Zusammengesetzt'}")
        print("  ", " | ".join([f"a={a}→{'✓' if res else '✗'}" for a, res in details]))
        if timings:
            times = [d["avg_time"] for d in timings.get("Miller-Rabin", []) if d["n"] == n]
            if times:
                print("   ", format_timing(times))

        # Solovay-Strassen
        from src.primality.tests import solovay_strassen_test
        result, details = solovay_strassen_test(n, k=3)
        print(f"Solovay-Strassen: {'✅ Prim' if result else '❌ Zusammengesetzt'}")
        print("  ", " | ".join([f"a={a}→{'✓' if res else '✗'}" for a, res in details]))
        if timings:
            times = [d["avg_time"] for d in timings.get("Solovay-Strassen", []) if d["n"] == n]
            if times:
                print("   ", format_timing(times))

        # AKS
        from src.primality.tests import aks_test
        result, steps = aks_test(n, return_steps=True)
        print(f"AKS: {'✅ Prim' if result else '❌ Zusammengesetzt'}")
        if steps:
            for step in steps:
                print(f"  {step}")
        if timings:
            times = [d["avg_time"] for d in timings.get("AKS", []) if d["n"] == n]
            if times:
                print("   ", format_timing(times))
