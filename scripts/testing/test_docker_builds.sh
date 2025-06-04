#!/bin/bash
# Docker build performance test script

echo "🚀 Starting Docker build performance test..."

# Test 1: Lightweight CI build
echo "📋 Test 1: Building lightweight CI image..."
time docker build -f docker/Dockerfile.ci -t insightspike-ai:ci-test . || {
    echo "❌ CI build failed"
    exit 1
}

echo "✅ Testing CI container..."
docker run --rm insightspike-ai:ci-test || {
    echo "❌ CI container test failed"
    exit 1
}

# Test 2: Production build with cache
echo "📋 Test 2: Building production image..."
time docker build -f docker/Dockerfile.main --target production -t insightspike-ai:prod-test . || {
    echo "❌ Production build failed"
    exit 1
}

echo "✅ Testing production container..."
docker run --rm insightspike-ai:prod-test python -c "import insightspike; print('Production Docker build successful')" || {
    echo "❌ Production container test failed"
    exit 1
}

# Test 3: Development build
echo "📋 Test 3: Building development image..."
time docker build -f docker/Dockerfile.main --target development -t insightspike-ai:dev-test . || {
    echo "❌ Development build failed"
    exit 1
}

echo "✅ Testing development container..."
docker run --rm insightspike-ai:dev-test python -c "import insightspike; print('Development Docker build successful')" || {
    echo "❌ Development container test failed"
    exit 1
}

echo "🎉 All Docker builds completed successfully!"

# Show image sizes
echo "📊 Image sizes:"
docker images | grep insightspike-ai

# Cleanup
echo "🧹 Cleaning up test images..."
docker rmi insightspike-ai:ci-test insightspike-ai:prod-test insightspike-ai:dev-test || true

echo "✨ Docker build performance test completed!"
