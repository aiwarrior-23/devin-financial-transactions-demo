def calculate_success_ratio(success_count, failure_count):
    total = success_count + failure_count
    
    ratio = success_count / total
    
    return {
        "total_requests": total,
        "success_ratio": ratio
    }