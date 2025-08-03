# 実験で使用したテストコーパス

## 実験結果サマリー

### 主要な発見
1. **距離ベース検索は明確な境界を提供**
   - Apple: 境界1.0で精度100%
   - Java: 分離度3.17x（最高）
   - Ruby gems: 分離度0.31x（予想通り曖昧）

2. **最適閾値の一貫性**
   - ほとんどのケースで距離1.0付近が境界
   - Python: 0.935
   - Java: 1.036
   - Spring: 1.047
   - Go: 1.093

3. **"Ruby gems"の興味深い挙動**
   - クエリ: "Ruby gems and libraries"
   - 最近傍: "Ruby is a precious red gemstone"（宝石）
   - プログラミングと宝石の境界が曖昧（意図的な多義性）

## Apple Query Test
```python
corpus = [
    # Apple company cluster
    "Apple Inc. makes innovative products.",
    "The iPhone is Apple's flagship device.", 
    "MacBook is a laptop made by Apple.",
    "Steve Jobs founded Apple Computer.",
    
    # Apple fruit cluster
    "I love eating fresh apples.",
    "Apple pie is a delicious dessert.",
    "Green apples are sour.",
    "An apple a day keeps the doctor away.",
    
    # Mixed/ambiguous
    "Apple's market share is growing.",  # Could be company or fruit market
    "The apple logo is iconic.",         # Company, but mentions fruit word
    
    # Control (unrelated)
    "Python is a programming language.",
    "The weather is nice today.",
]
```

## Ambiguous Query Test
```python
corpus = [
    # Python (programming)
    "Python is a versatile programming language.",
    "I write Python scripts for automation.",
    "Python has excellent data science libraries.",
    "Django is a Python web framework.",
    
    # Python (snake)
    "A python is a large constrictor snake.",
    "Pythons are found in tropical regions.",
    "The ball python is a popular pet snake.",
    "Pythons can grow over 20 feet long.",
    
    # Java (programming)
    "Java is an object-oriented programming language.",
    "Spring Boot simplifies Java development.",
    "Java runs on billions of devices worldwide.",
    "I'm learning Java for Android development.",
    
    # Java (island)
    "Java is the most populous island in Indonesia.",
    "Jakarta is the capital city on Java island.",
    "Java island has many active volcanoes.",
    "Coffee from Java is world-famous.",
    
    # Spring (framework)
    "Spring Framework is popular for enterprise Java.",
    "Spring Boot makes microservices easy.",
    "Spring Security handles authentication.",
    "I use Spring MVC for web applications.",
    
    # Spring (season)
    "Spring brings blooming flowers and warmer weather.",
    "In spring, the days get longer.",
    "Spring is between winter and summer.",
    "Cherry blossoms bloom in spring.",
    
    # Ruby (programming)
    "Ruby on Rails is a web framework.",
    "Ruby has elegant syntax.",
    "Many startups use Ruby for rapid development.",
    "Ruby is dynamically typed.",
    
    # Ruby (gemstone)
    "Ruby is a precious red gemstone.",
    "Rubies are valued for their deep red color.",
    "A ruby ring makes a beautiful gift.",
    "Rubies are harder than most gems.",
    
    # Go (programming)
    "Go is a language created by Google.",
    "Go has built-in concurrency support.",
    "Kubernetes is written in Go.",
    "Go compiles to native code.",
    
    # Go (game/verb)
    "Go is an ancient board game.",
    "Let's go to the park.",
    "Go ahead and start without me.",
    "The traffic light turned green, so go.",
]
```

## テストクエリ
1. "Apple products and technology"
2. "Python development and programming"
3. "Java enterprise applications"
4. "Spring framework configuration"
5. "Ruby gems and libraries"
6. "Go concurrent programming"

## 詳細な実験結果

### 各クエリの距離分布

#### 1. Apple products and technology
```
Company-related: 0.875 (±0.099)
Fruit-related:   1.087 (±0.035)
Gap: 0.212, Separation: 明確
```

#### 2. Python development and programming
```
Programming: 0.865 (±0.149)
Snake:       1.005 (±0.016)
Gap: 0.140, Separation: 1.70x
Winner: Cosine (1.75x)
```

#### 3. Java enterprise applications
```
Programming: 0.959 (±0.055)
Island:      1.113 (±0.042)
Gap: 0.154, Separation: 3.17x
Winner: Cosine (3.20x) - 最高の分離度！
```

#### 4. Spring framework configuration
```
Framework: 0.980 (±0.073)
Season:    1.115 (±0.058)
Gap: 0.135, Separation: 2.05x
Winner: Distance (2.05x vs 2.04x)
```

#### 5. Ruby gems and libraries
```
Programming: 0.957 (±0.061)
Gemstone:    0.984 (±0.120)
Gap: 0.027, Separation: 0.31x
Winner: Cosine (0.36x) - どちらも分離困難！

特筆事項：
- 最近傍が "Ruby is a precious red gemstone"
- "gems" の多義性により本質的に曖昧
```

#### 6. Go concurrent programming
```
Programming: 0.935 (±0.205)
Other:       1.251 (±0.117)
Gap: 0.316, Separation: 1.96x
Winner: Cosine (2.12x)
```

### 結論
- Sentence-BERTの正規化済みベクトルでは、距離1.0付近が意味的境界として機能
- 距離ベース検索は直感的で解釈しやすい閾値を提供
- 本質的に曖昧なクエリ（Ruby gems）も正しく検出される