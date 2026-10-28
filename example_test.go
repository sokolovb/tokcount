package tokcount_test

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/sokolovb/tokcount"
)

func Example() {
	// Load once at startup; a Counter is safe for concurrent use.
	qwen, err := tokcount.LoadHF("models/Qwen3.8-27B/tokenizer.json", nil)
	if err != nil {
		log.Fatal(err)
	}
	kimi, err := tokcount.LoadKimi("models/Kimi-K3", nil) // tiktoken.model + tokenizer_config.json
	if err != nil {
		log.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	text := "Hello, world!"
	n, err := qwen.Count(ctx, text)
	if err != nil {
		log.Fatal(err) // ctx.Err() if the deadline hit first
	}
	fmt.Println("qwen:", n)

	// Stop as soon as the answer is known.
	n, exceeded, err := kimi.CountUpTo(ctx, text, 8000)
	fmt.Println("kimi:", n, exceeded, err)
}
