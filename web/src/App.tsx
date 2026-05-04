import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import AskPage from "./pages/AskPage";
import SearchPage from "./pages/SearchPage";
import LibraryPage from "./pages/LibraryPage";
import IngestPage from "./pages/IngestPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/ask" replace />} />
        <Route path="/ask" element={<AskPage />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/library" element={<LibraryPage />} />
        <Route path="/ingest" element={<IngestPage />} />
      </Route>
    </Routes>
  );
}
