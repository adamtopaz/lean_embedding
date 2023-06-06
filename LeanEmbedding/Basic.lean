import Lean

open Lean

namespace Embedding

structure Config where
  apiKey : String

structure IndexedEmbedding where
  index : Nat
  embedding : Array JsonNumber
deriving ToJson, FromJson

structure Error where
  message : String
  type : String
deriving ToJson, FromJson

def Error.isTokenLimit (e : Error) : Bool := e.type == "invalid_request_error"
def Error.isServerError (e : Error) : Bool := e.type == "server error"

def parseResponse (response : String) :
    Except String (Except Error (Array IndexedEmbedding)) := do
  match Json.parse response with
  | .error e => .error s!"[Embedding.parseResponse]: Error parsing json:¬{e}\n{response}"
  | .ok json =>
    if let .ok error := json.getObjValAs? Error "error" then
      pure <| .error error
    else if let .ok data := json.getObjValAs? (Array IndexedEmbedding) "data" then
      pure <| .ok data
    else .error "[Embedding.parseResponse]: An error has occured.\n{response}"

end Embedding

abbrev EmbeddingM := ReaderT Embedding.Config IO

namespace EmbeddingM

open Embedding

def getRawResponse (data : Array String) : EmbeddingM (UInt32 × String × String) := do
  let child ← IO.Process.spawn {
    cmd := "curl"
    args := #[
      "https://api.openai.com/v1/embeddings",
      "-H", s!"Authorization: Bearer {(← read).apiKey}",
      "-H", "Content-Type: application/json",
      "--data-binary", "@-"
    ]
    stdin := .piped
    stdout := .piped
    stderr := .piped
  }
  let (stdin, child) ← child.takeStdin
  stdin.putStr <| toString <| toJson <| Json.mkObj [
    ("model","text-embedding-ada-002"),
    ("input", toString data)
  ]
  stdin.flush 
  let stdout ← IO.asTask child.stdout.readToEnd .dedicated
  let err ← child.stderr.readToEnd
  let exitCode ← child.wait
  if exitCode != 0 then 
    throw <| .userError err
  let out ← IO.ofExcept stdout.get
  return (exitCode, out, err)

def getIndexedEmbeddings (data : Array String) : EmbeddingM (Array IndexedEmbedding) := do
  let (_, rawResponse, _) ← getRawResponse data
  match parseResponse rawResponse with 
  | .ok (.ok out) => return out
  | .ok (.error err) => 
    if err.isServerError then 
      throw <| .userError s!"[EmbeddingM.getIndexedEmbeddings] Server error:\n{err.message}"
    else if err.isTokenLimit then 
      throw <| .userError s!"[EmbeddingM.getIndexedEmbeddings] Surpassed token limit:\n{err.message}"
    else throw <| .userError s!"[EmbeddingM.getIndexedEmbeddings] Unknown error:\n{toJson err}"
  | .error err => throw <| .userError s!"[EmbeddingM.getIndexedEmbeddings] Failed to parse response:\n{err}"

partial def getIndexedEmbeddingsRecursively (data : Array String) (gas : Nat := 5) (trace : Bool := false) : 
    EmbeddingM (Array IndexedEmbedding) := do
  if gas == 0 then 
    if trace then IO.println "[EmbeddingM.getEmbeddingsRecursively] Out of gas."
    return #[]
  if data.size == 0 then 
    if trace then IO.println "[EmbeddingM.getEmbeddingsRecursively] Empty input data."
    return #[]
  let (_, rawResponse, _) ← getRawResponse data
  match parseResponse rawResponse with 
  | .ok (.ok out) => 
    -- Supposedly the response should be sorted, but just in case...
    if trace then IO.println s!"[EmbeddingM.getEmbeddingsRecursively] Success with response of size {out.size}."
    return out
  | .ok (.error err) => 
    if err.isServerError then 
      if trace then IO.println s!"[EmbeddingM.getEmbeddingsRecursively] Server error. Retrying with gas = {gas-1}."
      getIndexedEmbeddingsRecursively data (gas - 1) trace
    else if err.isTokenLimit then 
      let size := data.size
      if size == 1 then 
        if trace then IO.println s!"[EmbeddingM.getEmbeddingsRecursively] Token limit reached with size 1. Ignoring."
        return #[]
      let newSize := size / 2
      if trace then IO.println s!"[EmbeddingM.getEmbeddingsRecursively] Token limit reached. Retrying with size {newSize}."
      let data1 := data[:newSize].toArray
      let data2 := data[newSize:].toArray
      return (← getIndexedEmbeddingsRecursively data1 gas trace) ++ (← getIndexedEmbeddingsRecursively data2 gas trace)
    else 
      if trace then IO.println s!"[EmbeddingM.getIndexedEmbedding] Unknown error. Ignoring."
      return #[]
  | .error err => 
      if trace then IO.println s!"[EmbeddingM.getIndexedEmbedding] Failed to parse response:\n{err}"
      return #[]

def runWith (m : EmbeddingM α) (apiKey : String) : IO α := do
  ReaderT.run m { apiKey := apiKey }

def run (m : EmbeddingM α) : IO α := do
  let some apiKey ← IO.getEnv "OPENAI_API_KEY" | 
    throw <| .userError "Failed to get OpenAI API Key."
  m.runWith apiKey

end EmbeddingM