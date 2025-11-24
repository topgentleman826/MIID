# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): YANEZ - MIID Team
# Copyright © 2025 YANEZ

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Name Variation Miner Module

This module implements a Bittensor miner that generates alternative spellings for names
using a local LLM (via Ollama). 
######### Ollama should be installed and running on the machine. ########
The miner receives requests from validators containing
a list of names and a query template, processes each name through the LLM, extracts
the variations from the LLM's response, and returns them to the validator.

The miner follows these steps:
1. Receive a request with names and a query template
2. For each name, query the LLM to generate variations
3. Process the LLM responses to extract clean variations
4. Return the variations to the validator

The processing logic handles different response formats from LLMs, including:
- Comma-separated lists
- Line-separated lists
- Space-separated lists with numbering

For debugging and analysis, the miner also saves:
- Raw LLM responses
- Processed variations in JSON format
- A pandas DataFrame with the variations

Each mining run is saved with a unique timestamp identifier to distinguish between
different runs and facilitate analysis of results over time.
"""

import json
import os
import re
import time
import typing
import traceback
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Set

import bittensor as bt
import jellyfish
import Levenshtein
import numpy as np
import ollama
import pandas as pd
from tqdm import tqdm

# Bittensor Miner Template:
from MIID.protocol import IdentitySynapse

# import base miner class which takes care of most of the boilerplate
from MIID.base.miner import BaseMinerNeuron

from bittensor.core.errors import NotVerifiedException


class Miner(BaseMinerNeuron):
    """
    Name Variation Miner Neuron
    
    This miner receives requests from validators to generate alternative spellings for names,
    and responds with variations generated using a local LLM (via Ollama).
    
    The miner handles the following tasks:
    - Processing incoming requests for name variations
    - Querying a local LLM to generate variations
    - Extracting and cleaning variations from LLM responses
    - Returning the processed variations to the validator
    - Saving intermediate results for debugging and analysis
    
    Each mining run is saved with a unique timestamp identifier to distinguish between
    different runs and facilitate analysis of results over time.
    
    Configuration:
    - model_name: The Ollama model to use (default: 'tinyllama:latest')
    - output_path: Directory for saving mining results (default: logging_dir/mining_results)
    """
    WHITELISTED_VALIDATORS = {
        "5C4qiYkqKjqGDSvzpf6YXCcnBgM6punh8BQJRP78bqMGsn54": "RoundTable21",
        "5DUB7kNLvvx8Dj7D8tn54N1C7Xok6GodNPQE2WECCaL9Wgpr": "Yanez", 
        "5GWzXSra6cBM337nuUU7YTjZQ6ewT2VakDpMj8Pw2i8v8PVs": "Yuma",
        "5HbUFHW4XVhbQvMbSy7WDjvhHb62nuYgP1XBsmmz9E2E2K6p": "OpenTensor",
        "5GQqAhLKVHRLpdTqRg1yc3xu7y47DicJykSpggE2GuDbfs54": "Rizzo",
        "5HK5tp6t2S59DywmHRWPBVJeJ86T61KjurYqeooqj8sREpeN": "Tensora",
        "5E2LP6EnZ54m3wS8s1yPvD5c3xo71kQroBw7aUVK32TKeZ5u": "Tao.bot",
        "5GuPvuyKBJAWQbEGAkMbfRpG5qDqqhML8uDVSWoFjqcKKvDU": "Testnet_omar",
        "5CnkkjPdfsA6jJDHv2U6QuiKiivDuvQpECC13ffdmSDbkgtt": "Testnet_asem"
    }

    def __init__(self, config=None):
        """
        Initialize the Name Variation Miner.
        
        Sets up the LLM client and creates directories for storing mining results.
        Each run will be saved in a separate directory with a unique timestamp.
        
        Args:
            config: Configuration object for the miner
        """
        super(Miner, self).__init__(config=config)
        
        self.model_name = getattr(self.config.neuron, 'model_name', None) if hasattr(self.config, 'neuron') else None
        if self.model_name is None:
            # Use llama3.1 for optimal balance of quality and speed (8B model)
            # This provides excellent phonetic/orthographic accuracy for scoring
            self.model_name = 'llama3.1:latest'
            bt.logging.info(f"No model specified in config, using default model: {self.model_name}")
            bt.logging.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            bt.logging.info("MODEL RECOMMENDATIONS FOR MAXIMUM ACCURACY:")
            bt.logging.info("  • llama3.1:latest (8B) - Default, excellent balance")
            bt.logging.info("  • llama3.3:latest (70B) - Best quality (requires 32GB+ VRAM)")
            bt.logging.info("  • qwen2.5:14b - Strong alternative, good multilingual")
            bt.logging.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        bt.logging.info(f"Using LLM model: {self.model_name}")
        bt.logging.info("Quality optimizations enabled: Few-shot prompting, Quality scoring, Phonetic ranking")

        # Configure output limits and client reuse for better resiliency
        self.max_variations = getattr(self.config.neuron, 'max_variations', 25)
        self.ollama_host = getattr(self.config.neuron, 'ollama_url', 'http://127.0.0.1:11434')
        self.cache_max_entries = getattr(self.config.neuron, 'response_cache_size', 128)
        self.target_variations = getattr(self.config.neuron, 'target_variations', self.max_variations)
        self._response_cache: OrderedDict[str, str] = OrderedDict()
        self.ollama_client = self._initialize_ollama_client()
        bt.logging.info(
            f"Configured to return up to {self.max_variations} variations per identity using Ollama host {self.ollama_host}"
        )
        
        # Create a directory for storing mining results
        # This helps with debugging and analysis
        self.output_path = os.path.join(self.config.logging.logging_dir, "mining_results")
        os.makedirs(self.output_path, exist_ok=True)
        bt.logging.info(f"Mining results will be saved to: {self.output_path}")
        self.axon.verify_fns[IdentitySynapse.__name__] = self._verify_validator_request

    def _initialize_ollama_client(self) -> ollama.Client:
        """Create and validate the Ollama client instance.

        Returns:
            ollama.Client: Configured client pointing at the configured host.

        Raises:
            RuntimeError: If Ollama is unavailable or the model cannot be pulled.
        """
        try:
            client = ollama.Client(host=self.ollama_host)
            models = client.list().get('models', [])
            model_exists = any(model.get('name') == self.model_name for model in models)

            if model_exists:
                bt.logging.info(f"Model {self.model_name} already pulled")
            else:
                bt.logging.info(f"Pulling model {self.model_name}...")
                client.pull(self.model_name)

            return client
        except Exception as e:
            bt.logging.error(f"Error with Ollama: {str(e)}")
            bt.logging.error("Make sure Ollama is installed and running on this machine")
            bt.logging.error("Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
            bt.logging.error("Start Ollama: ollama serve")
            raise RuntimeError("Ollama is required for this miner. Please install and start Ollama.")

    def _normalize_name(self, value: str) -> str:
        """Normalize names for consistent downstream comparison and caching."""
        normalized = unicodedata.normalize("NFKD", value)
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = re.sub(r"\s+", " ", normalized.strip())
        return normalized

    async def _verify_validator_request(self, synapse: IdentitySynapse) -> None:
        """
        Rejects any RPC that is not cryptographically proven to come from
        one of the whitelisted validator hotkeys.

        Signature *must* be present and valid.  If anything is missing or
        incorrect we raise `NotVerifiedException`, which the Axon middleware
        converts into a 401 reply.
        """
        # ----------  basic sanity checks  ----------
        if synapse.dendrite is None:
            raise NotVerifiedException("Missing dendrite terminal in request")

        hotkey    = synapse.dendrite.hotkey
        # signature = synapse.dendrite.signature
        nonce     = synapse.dendrite.nonce
        uuid      = synapse.dendrite.uuid
        body_hash = synapse.computed_body_hash

        # 1 — is the sender even on our allow‑list?
        if hotkey not in self.WHITELISTED_VALIDATORS:
            raise NotVerifiedException(f"{hotkey} is not a whitelisted validator")

        # 3 — run all the standard Bittensor checks (nonce window, replay,
        #     timeout, signature, …).  This *does not* insist on a signature,
        #     so we still do step 4 afterwards.
        message = (
            f"nonce: {nonce}. "
            f"hotkey {hotkey}. "
            f"self hotkey {self.wallet.hotkey.ss58_address}. "
            f"uuid {uuid}. "
            f"body hash {body_hash} "
        )
        bt.logging.info(
            f"Verifying message: {message}"
        )

        await self.axon.default_verify(synapse)

        # 5 — all good ➜ let the middleware continue
        bt.logging.info(
            f"Verified call from {self.WHITELISTED_VALIDATORS[hotkey]} ({hotkey})"
        )

    async def forward(self, synapse: IdentitySynapse) -> IdentitySynapse:
        """
        Process a name variation request by generating variations for each name.
        
        This is the main entry point for the miner's functionality. It:
        1. Receives a request with names and a query template
        2. Processes each name through the LLM
        3. Extracts variations from the LLM responses
        4. Returns the variations to the validator
        
        Each run is assigned a unique timestamp ID and results are saved in a
        dedicated directory for that run.
        
        Args:
            synapse: The IdentitySynapse containing names and query template
            
        Returns:
            The synapse with variations field populated with name variations
        """
        # Generate a unique run ID using timestamp
        run_id = int(time.time())
        bt.logging.info(f"Starting run {run_id} for {len(synapse.identity)} names")
        
        # Get timeout from synapse (default to 120s if not specified)
        timeout = getattr(synapse, 'timeout', 120.0)
        bt.logging.info(f"Request timeout: {timeout:.1f}s for {len(synapse.identity)} names")
        start_time = time.time()
        
        # Create a run-specific directory
        run_dir = os.path.join(self.output_path, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        
        # This will store all responses from the LLM in a format that can be processed later
        # Format: ["Respond", "---", "Query-{name}", "---", "{LLM response}"]
        Response_list = []
        
        # Track which names we've processed
        processed_names = []
        
        # Process each identity in the request, respecting the timeout
        for i, identity in enumerate(tqdm(synapse.identity, desc="Processing identities")):
            # Check if we're approaching the timeout (reserve 15% for processing)
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            time_buffer = timeout * 0.15  # Reserve 15% of total time for final processing
            
            # If time is running out, skip remaining identities
            if remaining < time_buffer:
                bt.logging.warning(
                    f"Time limit approaching ({elapsed:.1f}/{timeout:.1f}s), "
                    f"processed {len(processed_names)}/{len(synapse.identity)} identities. "
                    f"Skipping remaining identities to ensure timely response."
                )
                break
            
            # Extract name, dob, and address from identity array
            name = identity[0] if len(identity) > 0 else "Unknown"
            dob = identity[1] if len(identity) > 1 else "Unknown"
            address = identity[2] if len(identity) > 2 else "Unknown"
            
            # Format the response list for later processing
            Response_list.append("Respond")
            Response_list.append("---")
            Response_list.append("Query-" + name)
            Response_list.append("---")
            
            # Format the query with the current name, address, and DOB
            formatted_query = synapse.query_template.replace("{name}", name)
            formatted_query = formatted_query.replace("{address}", address)
            formatted_query = formatted_query.replace("{dob}", dob)
            
            # Query the LLM with timeout awareness
            try:
                bt.logging.info(f"Generating variations for name: {name}, remaining time: {remaining:.1f}s")
                # Pass a more limited timeout to the LLM call to ensure we stay within bounds
                name_respond = self.Get_Respond_LLM(formatted_query)
                Response_list.append(name_respond)
                processed_names.append(name)
            except Exception as e:
                bt.logging.error(f"Error querying LLM for name {name}: {str(e)}")
                Response_list.append("Error: " + str(e))
        
        # Process the responses to extract variations, but be aware of remaining time
        remaining = timeout - (time.time() - start_time)
        bt.logging.info(f"Processing responses with {remaining:.1f}s remaining of {timeout:.1f}s timeout")
        
        # Process variations even if we only got partial responses
        variations = {}
        if remaining > 1.0:  # Ensure at least 1 second for processing
            try:
                variations = self.process_variations(Response_list, run_id, run_dir, synapse.identity)
                bt.logging.info(f"Successfully processed variations for {len(variations)} names")
            except Exception as e:
                bt.logging.error(f"Error processing variations: {e}")
                bt.logging.error(f"Traceback: {traceback.format_exc()}")
                variations = {}
        
        # CRITICAL: Ensure we return variations for ALL requested names to maximize completeness score
        # Generate fallback variations for any missing names
        for identity in synapse.identity:
            if len(identity) > 0:
                name = identity[0]
                dob = identity[1] if len(identity) > 1 else "Unknown"
                address = identity[2] if len(identity) > 2 else "Unknown"
                
                if name not in variations or not variations[name]:
                    bt.logging.warning(f"Generating fallback variations for missing name: {name}")
                    variations[name] = self._generate_fallback_variations(name, dob, address, count=10)
        
        synapse.variations = variations
        
        # Log final timing information
        total_time = time.time() - start_time
        bt.logging.info(
            f"Request completed in {total_time:.2f}s of {timeout:.1f}s allowed. "
            f"Processed {len(processed_names)}/{len(synapse.identity)} names. "
            f"Returned variations for {len(variations)}/{len(synapse.identity)} names."
        )
        
        bt.logging.info(f"======== SYNAPSE VARIATIONS===============================================: {synapse.variations}")
        bt.logging.info(f"==========================Processed variations for {len(synapse.variations)} names in run {run_id}")
        bt.logging.info(f"==========================Synapse: {synapse}")
        bt.logging.info("========================================================================================")
        return synapse
    
    def Get_Respond_LLM(self, prompt: str) -> str:
        """
        Query the LLM using Ollama.
        
        This function sends a prompt to the LLM and returns its response.
        It uses the Ollama client to communicate with a locally running LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response as a string
            
        Raises:
            Exception: If there's an error communicating with the LLM
        """
        cached = self._get_cached_response(prompt)
        if cached is not None:
            bt.logging.trace("Reusing cached LLM response for prompt")
            return cached

        # Use Ollama to query the LLM
        try:
            # Reuse the initialized client to avoid per-call setup overhead
            response = self.ollama_client.chat(
                self.model_name,
                messages=self._build_llm_messages(prompt),
                options={
                    # Optimized for highest quality variations
                    "num_predict": 768,      # Increased for more detailed variations
                    "temperature": 0.65,     # Balanced: creative but controlled
                    "top_p": 0.9,           # Focused sampling for quality
                    "top_k": 50,            # Limit to top 50 tokens for consistency
                    "repeat_penalty": 1.2,   # Strong penalty against repetition
                    "frequency_penalty": 0.7, # Encourage diverse word choices
                    "presence_penalty": 0.6,  # Encourage new patterns
                }
            )

            # Extract and return the content of the response
            content = response['message']['content']
            self._cache_response(prompt, content)
            return content
        except Exception as e:
            bt.logging.error(f"LLM query failed: {str(e)}")
            raise

    def _build_llm_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Build high-quality prompts optimized for scoring metrics with few-shot examples."""
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert linguist specializing in name variation generation for identity security systems.\n\n"
                    "MISSION: Generate name variations that bypass detection while maintaining phonetic and visual similarity.\n\n"
                    "CRITICAL SCORING CRITERIA:\n"
                    "1. PHONETIC SIMILARITY (Highest Priority):\n"
                    "   - Variations must SOUND identical or nearly identical when spoken\n"
                    "   - Use Soundex/Metaphone-equivalent transformations\n"
                    "   - Maintain syllable structure and rhythm\n"
                    "\n"
                    "2. ORTHOGRAPHIC SIMILARITY:\n"
                    "   - Variations must LOOK similar to the original\n"
                    "   - Keep 60-80% of original letters\n"
                    "   - Maintain visual pattern recognition\n"
                    "\n"
                    "3. LENGTH CONSTRAINT (Critical):\n"
                    "   - MUST be within ±3 characters of original length\n"
                    "   - Prefer exact or ±1 character difference\n"
                    "\n"
                    "TRANSFORMATION RULES (Apply Multiple Per Variation):\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "Vowel Transformations:\n"
                    "  • a ↔ e, a ↔ ai, a ↔ ay\n"
                    "  • e ↔ i, e ↔ ee, e ↔ ea\n"
                    "  • i ↔ y, i ↔ ie, i ↔ ee\n"
                    "  • o ↔ u, o ↔ oo, o ↔ ow\n"
                    "  • u ↔ oo, u ↔ ou\n"
                    "\n"
                    "Consonant Transformations:\n"
                    "  • ph ↔ f ↔ v (phone → fone)\n"
                    "  • c ↔ k ↔ ck (carl → karl)\n"
                    "  • s ↔ z ↔ c (susan → zusan)\n"
                    "  • ch ↔ sh ↔ tch (charles → sharles)\n"
                    "  • j ↔ g ↔ dj (john → jon)\n"
                    "  • t ↔ tt ↔ th (smith → smyth)\n"
                    "  • gh → f → removed (laugh → laf)\n"
                    "\n"
                    "Double Letter Variations:\n"
                    "  • tt → t, nn → n, ll → l, ss → s\n"
                    "  • t → tt, n → nn, l → ll (reverse)\n"
                    "\n"
                    "Silent Letters:\n"
                    "  • Remove: h (john → jon), gh (night → nite)\n"
                    "  • Add: e at end (john → johne), h after consonants\n"
                    "\n"
                    "Cultural/Regional Variations:\n"
                    "  • British vs American spellings\n"
                    "  • Transliteration variants\n"
                    "  • Common typos that persist\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "EXAMPLES (Learn from these patterns):\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "Original: 'John Smith'\n"
                    "Variations:\n"
                    "  • Jon Smith (remove silent h)\n"
                    "  • John Smyth (i→y)\n"
                    "  • Jon Smyth (combine)\n"
                    "  • Jhon Smith (swap position)\n"
                    "  • John Smithe (add silent e)\n"
                    "  • Jahn Smith (o→a)\n"
                    "  • John Smitt (th→tt)\n"
                    "\n"
                    "Original: 'Mohammed'\n"
                    "Variations:\n"
                    "  • Muhammad (cultural variant)\n"
                    "  • Mohamed (remove d)\n"
                    "  • Mohammad (swap e/a)\n"
                    "  • Muhammed (add m)\n"
                    "  • Mohamad (remove e)\n"
                    "  • Muhamed (combine changes)\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "QUALITY CHECKLIST (Before Including Each Variation):\n"
                    "  ✓ Sounds similar when spoken?\n"
                    "  ✓ Looks similar visually?\n"
                    "  ✓ Within ±3 characters length?\n"
                    "  ✓ Uses 1-3 transformation rules?\n"
                    "  ✓ Realistic and plausible?\n"
                    "  ✓ Not a duplicate?\n"
                    "  ✓ Not identical to original?\n\n"
                    f"OUTPUT FORMAT:\n"
                    f"Generate EXACTLY {self.target_variations} high-quality variations (no fewer, no more).\n"
                    "Keep the same number of words as the source name unless a 1-word adjustment is critical for realism.\n"
                    "Keep the first and last character if possible, preserve capitalization, and avoid numbering/bullets.\n"
                    "Return ONLY valid JSON: {\"variations\": [\"variant1\", \"variant2\", ...]} with no trailing text."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    def _get_cached_response(self, prompt: str) -> Optional[str]:
        """Retrieve a cached LLM response if available."""
        cached = self._response_cache.get(prompt)
        if cached is not None:
            self._response_cache.move_to_end(prompt)
        return cached

    def _cache_response(self, prompt: str, response: str) -> None:
        """Store the response while enforcing an upper bound on cache size."""
        self._response_cache[prompt] = response
        self._response_cache.move_to_end(prompt)
        while len(self._response_cache) > self.cache_max_entries:
            self._response_cache.popitem(last=False)

    def _normalize_and_capitalize(self, seed: str, variation: str) -> str:
        """Normalize spacing and match capitalization to the seed."""
        normalized = self._normalize_name(variation)
        return self._apply_capitalization_pattern(seed, normalized)
    
    def process_variations(self, Response_list: List[str], run_id: int, run_dir: str, identity_list: List[List[str]]) -> Dict[str, List[List[str]]]:
        """
        Process LLM responses to extract identity variations.

        This function takes the raw LLM responses and extracts the name variations
        using the Process_function. It then creates structured variations that include
        name, DOB, and address variations for each identity.

        Args:
            Response_list: List of LLM responses in the format:
                          ["Respond", "---", "Query-{name}", "---", "{LLM response}"]
            run_id: Unique identifier for this processing run
            run_dir: Directory to save run-specific files
            identity_list: List of identity arrays, each containing [name, dob, address]

        Returns:
            Dictionary mapping each name to its list of [name, dob, address] variations
        """
        bt.logging.info(f"Processing {len(Response_list)} responses")
        Responds = "".join(Response_list).split("Respond")
        name_variations: Dict[str, List[List[str]]] = {}

        for i in range(1, len(Responds)):
            try:
                llm_respond = self.Process_function(Responds[i], False)
                name = llm_respond[0]

                matching_identity = None
                seed_name_lower = name.lower()
                for identity in identity_list:
                    if len(identity) > 0 and identity[0] and identity[0].strip().lower() == seed_name_lower:
                        matching_identity = identity
                        break

                if matching_identity is None:
                    bt.logging.warning(f"Could not find identity for name {name}")
                    continue

                seed_address = matching_identity[2] if len(matching_identity) > 2 else "Unknown"
                seed_dob = matching_identity[1] if len(matching_identity) > 1 else "Unknown"

                variations = [
                    var for var in llm_respond[2]
                    if not pd.isna(var) and var != "" and var.strip()
                ]
                variations = self._deduplicate_variations(variations)
                
                # Filter out variations that are too different in length (for better length score)
                variations = self._filter_by_length(variations, name)

                existing_variations = {seed_name_lower}
                existing_variations.update(var.lower() for var in variations)

                if len(variations) < self.max_variations:
                    needed = self.max_variations - len(variations)
                    heuristic_variations = self._generate_rule_based_variations(
                        name,
                        needed,
                        existing_variations,
                    )
                    if heuristic_variations:
                        bt.logging.debug(
                            f"Generated {len(heuristic_variations)} rule-based variations for {name} to reach target count"
                        )
                        variations.extend(heuristic_variations)

                if len(variations) > self.max_variations:
                    bt.logging.info(
                        f"Ranking {len(variations)} variations for {name} and selecting top {self.max_variations}"
                    )
                    variations = self._rank_and_filter_variations(variations, name, self.max_variations)
                elif variations:
                    # Even if we have fewer than max, still rank them for quality
                    variations = self._rank_and_filter_variations(variations, name, len(variations))

                structured_variations = []
                for var in variations:
                    cleaned_var = self._normalize_and_capitalize(name, var)

                    if cleaned_var and cleaned_var != name:  # Don't include exact match
                        # Keep DOB and address consistent with seed (validators check this)
                        structured_variation = [cleaned_var, seed_dob, seed_address]
                        structured_variations.append(structured_variation)

                structured_variations = self._pad_structured_variations(
                    name, seed_dob, seed_address, structured_variations
                )

                # Ensure we have enough variations (completeness score)
                if len(structured_variations) == 0:
                    bt.logging.warning(f"No valid variations generated for {name}, creating fallback variations")
                    structured_variations = self._generate_fallback_variations(name, seed_dob, seed_address)

                structured_variations = self._pad_structured_variations(
                    name, seed_dob, seed_address, structured_variations
                )

                name_variations[name] = structured_variations
                bt.logging.info(f"Processed {len(structured_variations)} variations for {name}")
            except Exception as e:
                bt.logging.error(f"Error processing response {i}: {e}")
                bt.logging.error(f"Traceback: {traceback.format_exc()}")

        bt.logging.info(f"Generated structured variations: {name_variations}")
        return name_variations
    
    def _filter_by_length(self, variations: List[str], original: str) -> List[str]:
        """Filter variations to keep only those with similar length to original (±3 chars for optimal score)."""
        original_len = len(original)
        filtered = []
        for var in variations:
            # Allow ±3 characters difference for optimal length score (tighter constraint)
            if abs(len(var) - original_len) <= 3:
                filtered.append(var)
            else:
                bt.logging.debug(f"Filtered out '{var}' due to length difference from '{original}'")
        if filtered:
            return filtered

        # If nothing passed the strict filter, fall back to closest length matches
        ranked_by_length = sorted(
            variations,
            key=lambda candidate: abs(len(candidate) - original_len)
        )
        return ranked_by_length[: self.max_variations]
    
    def _score_variation_quality(self, variation: str, original: str) -> float:
        """
        Score a variation based on phonetic and orthographic similarity.
        Returns a score between 0.0 and 1.0 (higher is better).
        
        This scoring function evaluates variations across multiple dimensions:
        - Length similarity (prefer close to original length)
        - Character overlap (Jaccard similarity)
        - Edit distance (Levenshtein)
        - Phonetic similarity (Soundex and Metaphone)
        """
        try:
            original_lower = original.lower()
            variation_lower = variation.lower()
            
            # 1. Length similarity (closer to original is better)
            length_diff = abs(len(variation) - len(original))
            length_score = max(0, 1.0 - (length_diff / max(len(original), 1)))
            
            # 2. Character overlap (Jaccard similarity)
            original_chars = set(original_lower)
            variation_chars = set(variation_lower)
            if original_chars:
                jaccard = len(original_chars & variation_chars) / len(original_chars | variation_chars)
            else:
                jaccard = 0.0
            
            # 3. Levenshtein distance (normalized edit distance)
            lev_dist = Levenshtein.distance(original_lower, variation_lower)
            lev_score = max(0, 1.0 - (lev_dist / max(len(original), len(variation))))
            
            # 4. Phonetic similarity (Soundex)
            try:
                soundex_orig = jellyfish.soundex(original)
                soundex_var = jellyfish.soundex(variation)
                soundex_match = 1.0 if soundex_orig == soundex_var else 0.5
            except:
                soundex_match = 0.5
            
            # 5. Phonetic similarity (Metaphone)
            try:
                metaphone_orig = jellyfish.metaphone(original)
                metaphone_var = jellyfish.metaphone(variation)
                metaphone_match = 1.0 if metaphone_orig == metaphone_var else 0.5
            except:
                metaphone_match = 0.5
            
            # Additional structure alignment: token count and edge characters
            original_parts = original_lower.split()
            variation_parts = variation_lower.split()
            part_delta = abs(len(original_parts) - len(variation_parts))
            structure_score = 1.0 if part_delta == 0 else 0.75 if part_delta == 1 else 0.5

            edge_score = 0.0
            if variation_lower:
                if variation_lower[0] == original_lower[:1]:
                    edge_score += 0.5
                if variation_lower[-1:] == original_lower[-1:]:
                    edge_score += 0.5

            # Combined score (weighted average optimized for validator scoring)
            quality_score = (
                length_score * 0.22 +       # Length similarity: 22%
                jaccard * 0.18 +            # Character overlap: 18%
                lev_score * 0.18 +          # Edit distance: 18%
                soundex_match * 0.16 +      # Phonetic (Soundex): 16%
                metaphone_match * 0.16 +    # Phonetic (Metaphone): 16%
                structure_score * 0.05 +    # Word-count alignment: 5%
                edge_score * 0.05           # Preserve first/last character: 5%
            )
            
            return quality_score
            
        except Exception as e:
            bt.logging.debug(f"Error scoring variation quality: {e}")
            return 0.5  # Return neutral score on error
    
    def _rank_and_filter_variations(self, variations: List[str], original: str, max_count: int) -> List[str]:
        """
        Rank variations by quality score and return the top N.
        This ensures only the highest quality variations are sent to validators.
        """
        if not variations:
            return []
        
        # Score each variation
        scored_variations = []
        for var in variations:
            score = self._score_variation_quality(var, original)
            scored_variations.append((var, score))
        
        # Sort by score (highest first)
        scored_variations.sort(key=lambda x: x[1], reverse=True)
        
        # Log the quality distribution
        if scored_variations:
            avg_score = sum(s[1] for s in scored_variations) / len(scored_variations)
            top_score = scored_variations[0][1]
            bt.logging.debug(f"Variation quality for '{original}': avg={avg_score:.3f}, top={top_score:.3f}")
        
        # Return top N variations
        return [var for var, score in scored_variations[:max_count]]

    def _apply_capitalization_pattern(self, template: str, variation: str) -> str:
        """Match the capitalization pattern of the template name."""
        template_words = template.split()
        variation_words = variation.split()
        formatted_words: List[str] = []

        for idx, word in enumerate(variation_words):
            template_word = template_words[idx] if idx < len(template_words) else template_words[-1] if template_words else ""
            if template_word.isupper():
                formatted_words.append(word.upper())
            elif template_word[:1].isupper():
                formatted_words.append(word[:1].upper() + word[1:])
            else:
                formatted_words.append(word.lower())

        return " ".join(formatted_words)

    def _generate_rule_based_variations(
        self,
        seed: str,
        count: int,
        existing_variations: Set[str],
    ) -> List[str]:
        """
        Generate additional variations using deterministic phonetic/orthographic rules.
        This is used to top-up the variation list when the LLM does not return enough items.
        """
        if count <= 0:
            return []

        variations: List[str] = []
        seed_lower = seed.lower()

        vowel_swaps = {
            "a": ["e", "ai", "ay"],
            "e": ["i", "ea", "ee"],
            "i": ["y", "ee"],
            "o": ["u", "oo", "oa"],
            "u": ["oo", "ou"],
            "y": ["i"],
        }

        pattern_replacements = {
            "ph": ["f", "v"],
            "ck": ["k"],
            "ch": ["sh", "k"],
            "c": ["k", "s"],
            "qu": ["kw"],
            "x": ["ks"],
            "ie": ["y"],
            "ee": ["i"],
            "oo": ["u"],
            "gh": ["f", ""],
        }

        double_letters = ["tt", "ll", "ss", "nn", "rr", "mm"]
        accent_map = {
            "a": ["á", "à", "â"],
            "e": ["é", "è", "ê"],
            "i": ["í", "ì"],
            "o": ["ó", "ò", "ô"],
            "u": ["ú", "ù"],
        }

        def add_variant(candidate: str) -> None:
            normalized = self._normalize_name(candidate)
            if not normalized:
                return

            normalized_lower = normalized.lower()
            if normalized_lower in existing_variations:
                return

            if abs(len(normalized) - len(seed)) > 3:
                return

            formatted = self._normalize_and_capitalize(seed, normalized)
            existing_variations.add(normalized_lower)
            variations.append(formatted)

        # 1. Vowel swaps
        for idx, char in enumerate(seed_lower):
            replacements = vowel_swaps.get(char, [])
            for repl in replacements:
                new_variant = seed_lower[:idx] + repl + seed_lower[idx + 1 :]
                add_variant(new_variant)
                if len(variations) >= count:
                    return variations

        # 2. Common pattern replacements
        for pattern, replacements in pattern_replacements.items():
            if pattern in seed_lower:
                for repl in replacements:
                    new_variant = seed_lower.replace(pattern, repl, 1)
                    add_variant(new_variant)
                    if len(variations) >= count:
                        return variations

        # 3. Double letter manipulations
        for dbl in double_letters:
            if dbl in seed_lower:
                new_variant = seed_lower.replace(dbl, dbl[0], 1)
                add_variant(new_variant)
                if len(variations) >= count:
                    return variations
            elif dbl[0] in seed_lower:
                new_variant = seed_lower.replace(dbl[0], dbl, 1)
                add_variant(new_variant)
                if len(variations) >= count:
                    return variations

        # 4. Accent/diacritic variations
        for idx, char in enumerate(seed_lower):
            replacements = accent_map.get(char, [])
            for repl in replacements:
                new_variant = seed_lower[:idx] + repl + seed_lower[idx + 1 :]
                add_variant(new_variant)
                if len(variations) >= count:
                    return variations

        # 5. Silent letter tweaks
        if "h" in seed_lower:
            new_variant = seed_lower.replace("h", "", 1)
            add_variant(new_variant)
        if not seed_lower.endswith("e"):
            add_variant(seed_lower + "e")
        else:
            add_variant(seed_lower[:-1])

        # 6. Swap first/last characters for short names
        if len(seed_lower.replace(" ", "")) >= 4:
            words = seed_lower.split()
            primary_word = words[0]
            if len(primary_word) > 3:
                swapped = primary_word[-1] + primary_word[1:-1] + primary_word[0]
                new_variant = " ".join([swapped] + words[1:])
                add_variant(new_variant)

        if len(variations) > count:
            ranked = self._rank_and_filter_variations(variations, seed, count)
            return ranked

        return variations
    
    def _generate_fallback_variations(self, name: str, dob: str, address: str, count: int = 5) -> List[List[str]]:
        """Generate deterministic fallback variations when the LLM fails."""
        existing_variations = {name.lower()}
        fallback_names = self._generate_rule_based_variations(
            name,
            max(count, self.max_variations),
            existing_variations,
        )

        if not fallback_names:
            fallback_names = [self._apply_capitalization_pattern(name, name)]

        structured: List[List[str]] = []
        for variant in fallback_names:
            structured.append([variant, dob, address])
            if len(structured) >= count:
                break

        return structured

    def _pad_structured_variations(
        self,
        seed_name: str,
        seed_dob: str,
        seed_address: str,
        structured_variations: List[List[str]],
    ) -> List[List[str]]:
        """Top up structured variations with deterministic rule-based candidates for better count/uniqueness scores."""
        target_count = min(self.target_variations, self.max_variations)

        if len(structured_variations) >= target_count:
            return structured_variations[:target_count]

        existing_variations = {seed_name.lower()}
        existing_variations.update(variation[0].lower() for variation in structured_variations)

        needed = target_count - len(structured_variations)
        additional_names = self._generate_rule_based_variations(
            seed_name,
            needed,
            existing_variations,
        )

        for variant in additional_names:
            if len(structured_variations) >= target_count:
                break

            structured_variations.append([variant, seed_dob, seed_address])

        return structured_variations
    
    def save_variations_to_json(self, name_variations: Dict[str, List[str]], run_id: int, run_dir: str) -> None:
        """
        Save processed variations to JSON and DataFrame for debugging and analysis.
        
        This function saves the processed variations in multiple formats:
        1. A pandas DataFrame saved as a pickle file in the run-specific directory
        2. A JSON file with the name variations in the run-specific directory
        3. A JSON file with the model name and run ID in the main output directory
        
        Each file is named with the run ID to distinguish between different runs.
        
        Args:
            name_variations: Dictionary mapping names to variations
            run_id: Unique identifier for this processing run
            run_dir: Directory to save run-specific files
        """
        bt.logging.info(f"=================== Name variations: {name_variations}")
        bt.logging.info(f"=================== Run ID: {run_id}")
        bt.logging.info(f"=================== Run directory: {run_dir}")
        bt.logging.info("Saving variations to JSON and DataFrame")

        # Find the maximum number of variations for any name
        max_variations = max([len(vars) for vars in name_variations.values()]) if name_variations else 0
        bt.logging.info(f"Maximum number of variations found: {max_variations}")
        
        # Create a DataFrame with columns for the name and each variation
        columns = ['Name'] + [f'Var_{i+1}' for i in range(max_variations)]
        result_df = pd.DataFrame(columns=columns)
        
        # Fill the DataFrame with names and their variations, padding with empty strings if needed
        for i, (name, variations) in enumerate(name_variations.items()):
            row_data = [name] + variations + [''] * (max_variations - len(variations))
            result_df.loc[i] = row_data
        
        # Note: We no longer need to clean the data here since it's already cleaned
        # in the process_variations function
        
        # Save DataFrame to pickle for backup and analysis
        # Include run_id in the filename
        #df_path = os.path.join(run_dir, f"variations_df_{run_id}.pkl")
        #result_df.to_pickle(df_path)
        
        # Convert DataFrame to JSON format
        json_data = {}
        for i, row in result_df.iterrows():
            name = row['Name']
            # Extract non-empty variations
            variations = [var for var in row[1:] if var != ""]
            json_data[name] = variations

        # Save to JSON file
        # Include run_id in the filename
        # json_path = os.path.join(run_dir, f"variations_{run_id}.json")
        # import json
        # with open(json_path, 'w', encoding='utf-8') as f:
        #     json.dump(json_data, f, indent=4)
        # bt.logging.info(f"Saved variations to: {json_path}")
        # bt.logging.info(f"DataFrame shape: {result_df.shape} with {max_variations} variation columns")

    def _deduplicate_variations(self, variations: List[str]) -> List[str]:
        """Return a list of unique variations while preserving order."""
        seen = set()
        deduped = []

        for variation in variations:
            normalized = variation.lower()
            if normalized in seen:
                continue

            seen.add(normalized)
            deduped.append(variation)

        return deduped
    
    def Clean_extra(self, payload: str, comma: bool, line: bool, space: bool, preserve_name_spaces: bool = False) -> str:
        """
        Clean the LLM output by removing unwanted characters.
        
        Args:
            payload: The text to clean
            comma: Whether to remove commas
            line: Whether to remove newlines
            space: Whether to remove spaces
            preserve_name_spaces: Whether to preserve spaces between names (for multi-part names)
        """
        # Remove punctuation and quotes
        payload = payload.replace(".", "")
        payload = payload.replace('"', "")
        payload = payload.replace("'", "")
        payload = payload.replace("-", "")
        payload = payload.replace("and ", "")
        
        # Handle spaces based on preservation flag
        if space:
            if preserve_name_spaces:
                # Replace multiple spaces with single space
                while "  " in payload:
                    payload = payload.replace("  ", " ")
            else:
                # Original behavior - remove all spaces
                payload = payload.replace(" ", "")
        
        if comma:
            payload = payload.replace(",", "")
        if line:
            payload = payload.replace("\\n", "")
        
        return payload.strip()

    def validate_variation(self, name: str, seed: str, is_multipart_name: bool) -> str:
        """
        Helper function to validate if a variation matches the seed name structure.
        
        Args:
            name: The variation to validate
            seed: The original seed name
            is_multipart_name: Whether the seed is a multi-part name
            
        Returns:
            str: The validated and cleaned variation, or np.nan if invalid
        """
        name = self._normalize_name(name)
        if not name or name.isspace():
            return np.nan

        # Remove common prefixes from LLM responses
        for prefix in ["Variation:", "Alt:", "Alternative:", "-", "*", "•"]:
            if name.startswith(prefix):
                name = name[len(prefix):].strip()
        
        # Reject names containing unexpected symbols to avoid penalties
        if not re.match(r"^[A-Za-z]+([ '\-][A-Za-z]+)*$", name):
            bt.logging.debug(f"Skipping variation '{name}' due to unexpected characters")
            return np.nan
        
        # Handle cases with colons (e.g., "Here are variations: Name")
        if ":" in name:
            name = name.split(":")[-1].strip()
        
        # Remove numbering (e.g., "1. Name" or "1) Name")
        name = re.sub(r'^\d+[\.\)]\s*', '', name)
        
        # Skip if empty after cleaning
        if not name or name.isspace():
            return np.nan
        
        # Skip if it's just the seed name (exact match)
        if name.lower() == seed.lower():
            return np.nan
        
        # Check length reasonability - be more lenient (±4 characters is acceptable)
        # This helps with length score
        if abs(len(name) - len(seed)) > 4:
            bt.logging.debug(f"Skipping variation '{name}' - length difference too large from '{seed}'")
            return np.nan
        
        # Check structure consistency with seed name (but be more lenient)
        name_parts = name.split()
        seed_parts = seed.split()
        
        if is_multipart_name:
            # For multi-part seed names (e.g., "John Smith"), variations should also have multiple parts
            # But allow some flexibility (e.g., 2-part seed can have 1-3 part variations)
            if len(name_parts) < 1:
                return np.nan
            # Be more lenient - allow variations with different number of parts if close
            if abs(len(name_parts) - len(seed_parts)) > 1:
                bt.logging.debug(f"Skipping variation '{name}' - part count too different from '{seed}'")
                return np.nan
        else:
            # For single-part seed names, prefer single-part variations but allow 2-part
            if len(name_parts) > 2:
                bt.logging.debug(f"Skipping multi-part variation '{name}' for single-part seed '{seed}'")
                return np.nan
            
        return name

    def Process_function(self, string: str, debug: bool) -> Tuple[str, str, List[str], Optional[str]]:
        """
        Process the LLM response to extract the seed name and variations.
        
        This function parses the LLM response to extract:
        1. The original seed name
        2. The list of name variations
        
        It handles different response formats from LLMs:
        - Comma-separated lists (preferred format)
        - Line-separated lists
        - Space-separated lists with numbering
        
        The function ensures variations match the structure of the seed name:
        - Single-part seed names (e.g., "John") only get single-part variations
        - Multi-part seed names (e.g., "John Smith") only get multi-part variations
        
        Args:
            string: The LLM response in the format:
                   "---\nQuery-{name}\n---\n{response}"
            debug: Whether to return debug information
            
        Returns:
            Tuple containing:
            - seed_name: The original name
            - processing_method: The method used to process the response (r1, r2, or r3)
            - variations_list: The list of extracted variations
            - payload: (if debug=True) The processed payload
        """
        # Split the response by "---" to extract the query and response parts
        splits = string.split('---')
        if len(splits) < 3 or "-" not in splits[1]:
            raise ValueError("Unexpected response format from LLM")

        # Extract and analyze the seed name structure
        seed = splits[1].split("-")[1].replace(".", "").replace(",", "").replace("'", "")
        seed_parts = seed.split()
        is_multipart_name = len(seed_parts) > 1
        seed = self.Clean_extra(seed, True, True, True, preserve_name_spaces=is_multipart_name)
        
        bt.logging.info(f"Processing seed name: '{seed}' (multipart: {is_multipart_name})")
        
        # Extract the response payload
        payload = splits[-1]

        json_result = self._try_parse_json_variations(payload, seed, is_multipart_name)
        if json_result is not None:
            parsed_seed, method, variations = json_result
            if debug:
                return parsed_seed, method, variations, payload
            return parsed_seed, method, variations
        
        # Case 1: Comma-separated list (preferred format)
        if len(payload.split(",")) > 3:  # Check if we have at least 3 commas
            # Clean the payload but keep commas for splitting
            payload = self.Clean_extra(payload, False, True, True, preserve_name_spaces=is_multipart_name)
            
            # Remove numbering prefixes
            for num in range(10):
                payload = payload.replace(str(num), "")
            
            # Split by comma and process each variation
            variations = []
            for name in payload.split(","):
                cleaned_var = self.validate_variation(name, seed, is_multipart_name)
                if not pd.isna(cleaned_var):
                    variations.append(cleaned_var)
            
            if debug:
                return seed, "r1", variations, payload
            return seed, "r1", variations
        
        # Case 2 & 3: Non-comma separated formats
        else:
            # Case 2: Line-separated list
            len_ans = len(payload.split("\\n"))
            if len_ans > 2:  # Multiple lines indicate line-separated format
                # Clean the payload but preserve newlines for splitting
                payload = self.Clean_extra(payload, True, False, True, preserve_name_spaces=is_multipart_name)
                
                # Remove numbering prefixes
                for num in range(10):
                    payload = payload.replace(str(num), "")
                
                # Process line-separated variations
                variations = []
                for name in payload.split("\\n"):
                    cleaned_var = self.validate_variation(name, seed, is_multipart_name)
                    if not pd.isna(cleaned_var):
                        variations.append(cleaned_var)
            
                if debug:
                    return seed, "r2", variations, payload
                return seed, "r2", variations
            
            # Case 3: Space-separated list
            else:
                # Clean the payload but preserve spaces for multi-part names
                payload = self.Clean_extra(payload, True, True, False, preserve_name_spaces=is_multipart_name)
                
                # Remove numbering prefixes
                for num in range(10):
                    payload = payload.replace(str(num), "")
                
                variations = []
                if is_multipart_name:
                    # For multi-part names, we need to carefully group the parts
                    current_variation = []
                    parts = payload.split()
                    
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        
                        if ":" in part:  # New variation starts after colon
                            if current_variation:
                                # Process completed variation
                                cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name)
                                if not pd.isna(cleaned_var):
                                    variations.append(cleaned_var)
                            current_variation = [part.split(":")[-1].strip()]
                        else:
                            current_variation.append(part)
                            # Check if we have collected enough parts for a complete name
                            if len(current_variation) == len(seed_parts):
                                cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name)
                                if not pd.isna(cleaned_var):
                                    variations.append(cleaned_var)
                                current_variation = []
                
                    # Handle any remaining parts
                    if current_variation:
                        cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name)
                        if not pd.isna(cleaned_var):
                            variations.append(cleaned_var)
                else:
                    # For single-part names, simple space splitting is sufficient
                    for name in payload.split():
                        cleaned_var = self.validate_variation(name, seed, is_multipart_name)
                        if not pd.isna(cleaned_var):
                            variations.append(cleaned_var)
                
            if debug:
                return seed, "r3", variations, payload
            return seed, "r3", variations

    def _try_parse_json_variations(
        self, payload: str, seed: str, is_multipart_name: bool
    ) -> Optional[Tuple[str, str, List[str]]]:
        """Quickly parse structured JSON responses for lower latency processing."""
        try:
            cleaned_payload = payload.strip()
            
            # Try multiple extraction strategies
            # Strategy 1: Extract from markdown code blocks
            if "```" in cleaned_payload:
                # Try json code block first
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', cleaned_payload, re.DOTALL)
                if json_match:
                    cleaned_payload = json_match.group(1)
                else:
                    # Try plain code block
                    code_match = re.search(r'```\s*(\{.*?\})\s*```', cleaned_payload, re.DOTALL)
                    if code_match:
                        cleaned_payload = code_match.group(1)
            
            # Strategy 2: Find JSON object directly
            if not cleaned_payload.strip().startswith('{'):
                json_obj_match = re.search(r'\{[^{}]*"variations"[^{}]*\[[^\]]*\][^{}]*\}', cleaned_payload, re.DOTALL)
                if json_obj_match:
                    cleaned_payload = json_obj_match.group(0)

            data = json.loads(cleaned_payload)
            variations_field = data.get("variations") if isinstance(data, dict) else None
            if not isinstance(variations_field, list):
                bt.logging.debug("No 'variations' list found in JSON response")
                return None

            variations: List[str] = []
            for entry in variations_field:
                # Handle both string entries and dict entries
                if isinstance(entry, dict):
                    # Try common keys that might contain the variation
                    variation_str = entry.get('name') or entry.get('variation') or entry.get('value') or str(entry)
                else:
                    variation_str = str(entry)
                
                cleaned_var = self.validate_variation(variation_str, seed, is_multipart_name)
                if not pd.isna(cleaned_var) and cleaned_var.strip():
                    variations.append(cleaned_var)

            if variations:
                bt.logging.debug(f"Successfully extracted {len(variations)} variations from JSON")
                return seed, "json", variations
            else:
                bt.logging.debug("No valid variations found in JSON response")
                return None
                
        except json.JSONDecodeError as exc:
            bt.logging.trace(f"JSON parsing failed: {exc}")
            return None
        except Exception as exc:
            bt.logging.trace(f"JSON parsing error, falling back to heuristic parser: {exc}")
            return None

    async def blacklist(
        self, synapse: IdentitySynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored.
        
        This function implements security checks to ensure that only authorized
        validators can query this miner. It verifies:
        1. Whether the request has a valid dendrite and hotkey
        2. Whether the hotkey is one of the ones on the white list
        
        Args:
            synapse: A IdentitySynapse object constructed from the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing:
                - bool: Whether the request should be blacklisted
                - str: The reason for the decision
        """
        # Check if the request has a valid dendrite and hotkey
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        if synapse.dendrite.hotkey not in self.WHITELISTED_VALIDATORS:
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        # If all checks pass, allow the request
        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: IdentitySynapse) -> float:
        """
        The priority function determines the order in which requests are handled.
        
        This function assigns a priority to each request based on the stake of the
        calling entity. Requests with higher priority are processed first, which
        ensures that validators with more stake get faster responses.
        
        Args:
            synapse: The IdentitySynapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.
                  Higher values indicate higher priority.
        """
        # Check if the request has a valid dendrite and hotkey
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        # Get the UID of the caller
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )
        
        # Use the stake as the priority
        # Higher stake = higher priority
        priority = float(
            self.metagraph.S[caller_uid]
        )
        
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"----------------------------------Name Variation Miner running... {time.time()}")
            time.sleep(30)
