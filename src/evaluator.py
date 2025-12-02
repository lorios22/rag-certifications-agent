import time
import logging
from typing import List, Dict, Any
from datetime import datetime
import json

from .rag_agent import RAGAgent
from .config import EVALUATION_QUESTIONS, LOGS_DIR

logging.basicConfig(
    filename=LOGS_DIR / "evaluation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(self, agent: RAGAgent):
        self.agent = agent
        self.evaluation_results = []

    def evaluate_questions(self, questions: List[str] = None) -> Dict[str, Any]:
        if questions is None:
            questions = EVALUATION_QUESTIONS
        
        logger.info(f"Iniciando evaluación con {len(questions)} preguntas")
        
        results = []
        total_time = 0
        successful = 0
        failed = 0
        
        for i, question in enumerate(questions, 1):
            logger.info(f"Evaluando pregunta {i}/{len(questions)}: {question}")
            
            start_time = time.time()
            response = self.agent.ask(question)
            response_time = time.time() - start_time
            
            total_time += response_time
            
            if response["success"]:
                successful += 1
            else:
                failed += 1
            
            result = {
                "question": question,
                "answer": response["answer"],
                "success": response["success"],
                "response_time": response_time,
                "num_sources": response.get("metadata", {}).get("num_sources", 0),
                "timestamp": str(datetime.now())
            }
            
            results.append(result)
            self.evaluation_results.append(result)
        
        evaluation_summary = {
            "total_questions": len(questions),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(questions) * 100,
            "avg_response_time": total_time / len(questions),
            "total_time": total_time,
            "results": results,
            "timestamp": str(datetime.now())
        }
        
        logger.info(f"Evaluación completada. Tasa de éxito: {evaluation_summary['success_rate']:.2f}%")
        
        return evaluation_summary

    def evaluate_single_question(self, question: str, expected_answer: str = None) -> Dict[str, Any]:
        start_time = time.time()
        response = self.agent.ask(question)
        response_time = time.time() - start_time
        
        result = {
            "question": question,
            "answer": response["answer"],
            "expected_answer": expected_answer,
            "success": response["success"],
            "response_time": response_time,
            "metadata": response.get("metadata", {}),
            "timestamp": str(datetime.now())
        }
        
        if expected_answer:
            result["answer_matches"] = self._compare_answers(response["answer"], expected_answer)
        
        self.evaluation_results.append(result)
        
        return result

    def _compare_answers(self, answer: str, expected: str) -> bool:
        answer_lower = answer.lower().strip()
        expected_lower = expected.lower().strip()
        
        return expected_lower in answer_lower or answer_lower in expected_lower

    def get_evaluation_report(self) -> str:
        if not self.evaluation_results:
            return "No hay resultados de evaluación disponibles"
        
        report = "Reporte de Evaluación\n"
        report += "=" * 50 + "\n\n"
        
        total = len(self.evaluation_results)
        successful = sum(1 for r in self.evaluation_results if r["success"])
        avg_time = sum(r["response_time"] for r in self.evaluation_results) / total
        
        report += f"Total de preguntas: {total}\n"
        report += f"Exitosas: {successful}\n"
        report += f"Fallidas: {total - successful}\n"
        report += f"Tasa de éxito: {successful/total*100:.2f}%\n"
        report += f"Tiempo promedio: {avg_time:.2f}s\n\n"
        
        report += "Detalles por pregunta:\n"
        report += "-" * 50 + "\n"
        
        for i, result in enumerate(self.evaluation_results, 1):
            report += f"\n{i}. {result['question']}\n"
            report += f"   Respuesta: {result['answer'][:100]}...\n"
            report += f"   Tiempo: {result['response_time']:.2f}s\n"
            report += f"   Estado: {'Exitosa' if result['success'] else 'Fallida'}\n"
        
        return report

    def save_results(self, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = LOGS_DIR / f"evaluation_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Resultados guardados en {filename}")
        
        return filename

    def measure_retrieval_quality(self, questions: List[str] = None) -> Dict[str, Any]:
        if questions is None:
            questions = EVALUATION_QUESTIONS[:5]
        
        retrieval_stats = {
            "avg_sources": 0,
            "min_sources": float('inf'),
            "max_sources": 0,
            "results": []
        }
        
        total_sources = 0
        
        for question in questions:
            response = self.agent.ask(question)
            num_sources = response.get("metadata", {}).get("num_sources", 0)
            
            total_sources += num_sources
            retrieval_stats["min_sources"] = min(retrieval_stats["min_sources"], num_sources)
            retrieval_stats["max_sources"] = max(retrieval_stats["max_sources"], num_sources)
            
            retrieval_stats["results"].append({
                "question": question,
                "num_sources": num_sources,
                "sources": response.get("metadata", {}).get("sources", [])
            })
        
        retrieval_stats["avg_sources"] = total_sources / len(questions)
        
        return retrieval_stats

    def measure_response_time(self, iterations: int = 10) -> Dict[str, Any]:
        question = "¿Qué es la certificación AWS Machine Learning?"
        
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            self.agent.ask(question)
            response_time = time.time() - start_time
            times.append(response_time)
        
        return {
            "iterations": iterations,
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "total_time": sum(times),
            "times": times
        }
