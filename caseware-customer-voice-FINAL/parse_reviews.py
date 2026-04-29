"""
Parse raw review text files from Capterra, G2, Reddit, and Software Advice
into a structured CSV dataset for NLP analysis.
"""
import re
import csv
import os

def parse_capterra(filepath):
    """Parse Capterra reviews."""
    reviews = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Split by reviewer blocks - look for patterns like "Name\nRole in Country"
    # Capterra format: Name, Role, Company info, date, rating, then Pros/Cons
    
    # Find all review blocks using the rating pattern
    blocks = re.split(r'\n(?=[A-Z][a-z]+\r?\n)', text)
    
    current_review = {}
    lines = text.replace('\r', '').split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for rating pattern like "5.0" or "4.0" on its own line after a date
        if re.match(r'^[1-5]\.\d$', line):
            if current_review.get('text'):
                reviews.append(current_review)
            current_review = {'rating': float(line), 'source': 'Capterra', 'text': '', 'pros': '', 'cons': ''}
        
        elif line == 'Pros:':
            i += 1
            pros_text = []
            while i < len(lines) and lines[i].strip() != 'Cons:' and lines[i].strip() != 'Reviewer Source':
                if lines[i].strip():
                    pros_text.append(lines[i].strip())
                i += 1
            current_review['pros'] = ' '.join(pros_text)
            continue
            
        elif line == 'Cons:':
            i += 1
            cons_text = []
            while i < len(lines) and lines[i].strip() != 'Reviewer Source' and lines[i].strip() != 'Show more details' and not re.match(r'^[A-Z][a-z]+$', lines[i].strip()):
                if lines[i].strip() and lines[i].strip() not in ['Show more details', 'Reviewer Source']:
                    cons_text.append(lines[i].strip())
                i += 1
            current_review['cons'] = ' '.join(cons_text)
            current_review['text'] = f"{current_review.get('pros', '')} {current_review.get('cons', '')}".strip()
            continue
        
        i += 1
    
    if current_review.get('text'):
        reviews.append(current_review)
    
    return reviews


def parse_g2(filepath):
    """Parse G2 reviews."""
    reviews = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read().replace('\r', '')
    
    lines = text.split('\n')
    i = 0
    current_review = {}
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Rating pattern: "5/5" or "4.5/5"
        if re.match(r'^[0-9]+\.?[0-9]*/5$', line) and i + 1 < len(lines) and 'What do you like best' in lines[i+1]:
            rating_str = line.replace('/5', '')
            try:
                rating = float(rating_str)
                # Normalize to 5-point scale
                current_review = {'rating': rating, 'source': 'G2', 'pros': '', 'cons': '', 'text': ''}
            except:
                pass
        
        elif 'What do you like best' in line:
            i += 1
            pros_text = []
            while i < len(lines) and 'What do you dislike' not in lines[i]:
                if lines[i].strip() and lines[i].strip() not in ['Show More', 'Show Less']:
                    pros_text.append(lines[i].strip())
                i += 1
            current_review['pros'] = ' '.join(pros_text)
            continue
        
        elif 'What do you dislike' in line:
            i += 1
            cons_text = []
            while i < len(lines) and not re.match(r'^[0-9]+\.?[0-9]*/5$', lines[i].strip()) and 'What do you like best' not in lines[i] and 'Recommendations to others' not in lines[i] and 'What problems' not in lines[i]:
                if lines[i].strip() and lines[i].strip() not in ['Show More', 'Show Less', 'Response from Caseware']:
                    cons_text.append(lines[i].strip())
                i += 1
            current_review['cons'] = ' '.join(cons_text)
            current_review['text'] = f"{current_review.get('pros', '')} {current_review.get('cons', '')}".strip()
            if current_review.get('text') and len(current_review['text']) > 10:
                reviews.append(current_review.copy())
            continue
        
        i += 1
    
    return reviews


def parse_software_advice(filepath):
    """Parse Software Advice reviews."""
    reviews = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read().replace('\r', '')
    
    lines = text.split('\n')
    i = 0
    current_review = {}
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line == 'Pros:':
            i += 1
            pros_text = []
            while i < len(lines) and lines[i].strip() != 'Cons:':
                if lines[i].strip() and lines[i].strip() not in ['Show more', 'Show less']:
                    pros_text.append(lines[i].strip())
                i += 1
            current_review['pros'] = ' '.join(pros_text)
            current_review['source'] = 'Software Advice'
            continue
        
        elif line == 'Cons:':
            i += 1
            cons_text = []
            while i < len(lines) and lines[i].strip() != 'Reviewer' and lines[i].strip() != 'Ratings Breakdown' and lines[i].strip() != 'Pros:':
                if lines[i].strip() and lines[i].strip() not in ['Show more', 'Show less']:
                    cons_text.append(lines[i].strip())
                i += 1
            current_review['cons'] = ' '.join(cons_text)
            current_review['text'] = f"{current_review.get('pros', '')} {current_review.get('cons', '')}".strip()
            if current_review.get('text') and len(current_review['text']) > 10:
                current_review.setdefault('rating', 0)
                reviews.append(current_review.copy())
                current_review = {}
            continue
        
        # Try to find rating - "Overall: X.X / 5" or standalone numbers near "Ratings Breakdown"
        rating_match = re.match(r'^(\d+\.?\d*)\s*/\s*5', line)
        if rating_match:
            try:
                current_review['rating'] = float(rating_match.group(1))
            except:
                pass
        
        i += 1
    
    return reviews


def parse_reddit(filepath):
    """Parse Reddit threads — filter to Caseware-relevant comments only."""
    reviews = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read().replace('\r', '')
    
    lines = text.split('\n')
    
    # Caseware-related keywords for relevance filtering
    cw_keywords = [
        'caseware', 'working paper', 'workpaper', 'das ', 'onpoint',
        'cloud version', 'desktop version', 'trial balance', 'lead sheet',
        'engagement software', 'audit software', 'cw ', 'cloudbridge',
        'working papers', 'case ware'
    ]
    
    skip_patterns = [
        'Skip to', 'Search in', 'Advertise', 'Open chat', 'Create post',
        'Open inbox', 'Expand user', 'Go to', 'Back', 'avatar', 'Upvote',
        'Downvote', 'Share', 'Reply', 'Award', 'Sort by', 'Best', 'New',
        'Join the conversation', 'Go to comments', 'Promoted',
        'u/', 'r/', 'Learn More', 'Thumbnail', 'questrade',
        'Search Comments', 'Expand comment', 'Comments Section', 'more replies',
        'Comment deleted', 'level ', 'Continue this thread', 'Privacy Policy',
        'Report', 'Save', 'Hide', 'Block', 'Follow', 'Settings', 'Log In',
        'Sign Up', 'Get the app', 'Terms', 'Content Policy', 'Reddit, Inc',
        'Show parent', 'Collapse thread', 'sidebar promoted', 'thumbnail image',
        'Live chat on Discord', 'No self-promotion', 'Recruiting Guide',
        'Busy Season Tips', 'Podcast', 'Growing Need', 'Excel Exposure',
        'Compensation Threads', 'StartHereGoPlaces', 'ThisWaytoCPA',
        'click here for an invite', 'post thumbnail', 'Primarily for accountants'
    ]
    
    comments = []
    current_comment = []
    
    for line in lines:
        line = line.strip()
        
        if not line or len(line) < 15:
            if current_comment:
                text = ' '.join(current_comment)
                if len(text) > 40:
                    comments.append(text)
                current_comment = []
            continue
        
        if any(skip in line for skip in skip_patterns):
            continue
        
        if re.match(r'^\d+$', line):
            continue
        
        current_comment.append(line)
    
    if current_comment:
        text = ' '.join(current_comment)
        if len(text) > 40:
            comments.append(text)
    
    # Filter to Caseware-relevant comments only
    for comment in comments:
        if any(kw in comment.lower() for kw in cw_keywords):
            reviews.append({
                'source': 'Reddit',
                'rating': 0,
                'text': comment,
                'pros': '',
                'cons': ''
            })
    
    return reviews


if __name__ == '__main__':
    all_reviews = []
    
    capterra = parse_capterra('/mnt/user-data/uploads/capterra.txt')
    print(f"Capterra: {len(capterra)} reviews")
    all_reviews.extend(capterra)
    
    g2 = parse_g2('/mnt/user-data/uploads/G2.txt')
    print(f"G2: {len(g2)} reviews")
    all_reviews.extend(g2)
    
    sa = parse_software_advice('/mnt/user-data/uploads/software_advice.txt')
    print(f"Software Advice: {len(sa)} reviews")
    all_reviews.extend(sa)
    
    reddit = parse_reddit('/mnt/user-data/uploads/reddit.txt')
    print(f"Reddit: {len(reddit)} comments")
    all_reviews.extend(reddit)
    
    print(f"\nTotal: {len(all_reviews)} reviews/comments")
    
    # Write to CSV
    output_path = '/home/claude/caseware-sentiment/reviews.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'rating', 'text', 'pros', 'cons'])
        writer.writeheader()
        for review in all_reviews:
            writer.writerow(review)
    
    print(f"Saved to {output_path}")
    
    # Print samples
    for source in ['Capterra', 'G2', 'Software Advice', 'Reddit']:
        source_reviews = [r for r in all_reviews if r['source'] == source]
        if source_reviews:
            print(f"\n--- {source} Sample ---")
            sample = source_reviews[0]
            print(f"Rating: {sample['rating']}")
            print(f"Text: {sample['text'][:150]}...")
