#!/bin/bash
# Record InsightSpike demo using asciinema and convert to GIF

echo "🎬 Recording InsightSpike-AI Demo..."
echo "===================================="

# Check if asciinema is installed
if ! command -v asciinema &> /dev/null; then
    echo "⚠️  asciinema not found. Installing..."
    if command -v brew &> /dev/null; then
        brew install asciinema
    else
        echo "❌ Please install asciinema first:"
        echo "   macOS: brew install asciinema"
        echo "   Linux: sudo apt-get install asciinema"
        exit 1
    fi
fi

# Check if agg (asciinema gif generator) is installed
if ! command -v agg &> /dev/null; then
    echo "⚠️  agg not found. Installing..."
    if command -v brew &> /dev/null; then
        brew install agg
    else
        echo "❌ Please install agg first:"
        echo "   macOS: brew install agg"
        echo "   Linux: cargo install --git https://github.com/asciinema/agg"
        exit 1
    fi
fi

# Record the demo
echo ""
echo "📹 Starting recording in 3 seconds..."
echo "   Press Ctrl+D when the demo is complete"
sleep 3

# Record with asciinema
asciinema rec --overwrite demo.cast -c "poetry run python scripts/create_demo.py"

# Convert to GIF
echo ""
echo "🎨 Converting to GIF..."
agg demo.cast demo.gif \
    --font-family "Monaco,Menlo,monospace" \
    --font-size 14 \
    --theme monokai \
    --speed 1.5

# Clean up
rm demo.cast

echo ""
echo "✅ Demo GIF created: demo.gif"
echo "📏 Size: $(ls -lh demo.gif | awk '{print $5}')"
echo ""
echo "📝 To add to README:"
echo "   ![InsightSpike Demo](demo.gif)"