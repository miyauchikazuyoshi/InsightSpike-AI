#!/bin/bash
# Prepare InsightSpike-AI for PyPI release

echo "📦 Preparing InsightSpike-AI for PyPI release..."
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this script from the project root."
    exit 1
fi

# 1. Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info
echo "✅ Cleaned"

# 2. Check pyproject.toml
echo ""
echo "📋 Checking pyproject.toml metadata..."
poetry check
if [ $? -ne 0 ]; then
    echo "❌ pyproject.toml validation failed"
    exit 1
fi
echo "✅ pyproject.toml is valid"

# 3. Build the package
echo ""
echo "🔨 Building package..."
poetry build
if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi
echo "✅ Package built successfully"

# 4. Check the built files
echo ""
echo "📦 Built files:"
ls -la dist/

# 5. Test the package locally
echo ""
echo "🧪 Testing package installation..."
python -m venv test_env
source test_env/bin/activate
pip install dist/*.whl
python -c "import insightspike; print('✅ Package imports successfully')"
deactivate
rm -rf test_env

# 6. Dry run for PyPI
echo ""
echo "🚀 Testing PyPI upload (dry run)..."
poetry publish --dry-run
if [ $? -ne 0 ]; then
    echo "❌ Dry run failed"
    exit 1
fi
echo "✅ Dry run successful"

# 7. Create release checklist
echo ""
echo "📝 Creating release checklist..."
cat > PYPI_RELEASE_CHECKLIST.md << EOF
# PyPI Release Checklist

## Pre-release
- [ ] Update version in pyproject.toml
- [ ] Update CHANGELOG.md
- [ ] Run all tests: \`poetry run pytest\`
- [ ] Check documentation is up to date
- [ ] Tag the release: \`git tag -a v0.8.0 -m "Release v0.8.0"\`

## Release
- [ ] Build package: \`poetry build\`
- [ ] Upload to TestPyPI first: \`poetry publish -r testpypi\`
- [ ] Test installation from TestPyPI
- [ ] Upload to PyPI: \`poetry publish\`

## Post-release
- [ ] Push tags: \`git push --tags\`
- [ ] Create GitHub release
- [ ] Announce on social media
- [ ] Update documentation site

## Rollback (if needed)
- [ ] Yank package: \`pip install -U twine && twine yank insightspike-ai==0.8.0\`
EOF

echo "✅ Created PYPI_RELEASE_CHECKLIST.md"

# Summary
echo ""
echo "🎉 PyPI preparation complete!"
echo "============================="
echo "✅ Package builds successfully"
echo "✅ Metadata validation passed"
echo "✅ Dry run successful"
echo ""
echo "📋 Next steps:"
echo "1. Review PYPI_RELEASE_CHECKLIST.md"
echo "2. Set up PyPI credentials with: poetry config pypi-token.pypi <your-token>"
echo "3. When ready, run: poetry publish"