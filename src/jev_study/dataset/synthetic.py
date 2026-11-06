"""Placeholder sintético. Os rótulos foram escritos para este estudo, não medidos.

Box entra só como skip: a pergunta existe no schema, mas sem imagem o runner
não chama o backend. Estados e instruções estão em português; as chaves, em inglês.
"""

from jev_study.schema import (
    BoxQuestion,
    ChoiceQuestion,
    NoulCriteria,
    NoulQuestion,
    Question,
    ScoreQuestion,
)
from jev_study.tasks import DecisionTask, ExpectedLabel


def load_synthetic_tasks() -> list[DecisionTask]:
    """Devolve a lista fechada do PR1, com id e material únicos."""
    tasks = _build()
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("id de tarefa duplicado")
    fingerprints = [task.fingerprint() for task in tasks if not task.skip]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("tarefas com o mesmo estado e as mesmas perguntas")
    for task in tasks:
        if not task.skip and not task.note.strip():
            raise ValueError(f"{task.id} sem justificativa do gabarito")
    return tasks


def _build() -> list[DecisionTask]:
    return [
        _labeled(
            "route_duplicate_charge",
            "Fui cobrado duas vezes na assinatura deste mês. Quero o estorno da cobrança extra.",
            "team",
            _route(),
            ExpectedLabel(choice="billing"),
            "O pedido é estorno de cobrança. Não há defeito de produto nem mudança cadastral.",
        ),
        _labeled(
            "route_app_crash",
            "O aplicativo fecha sozinho ao abrir o relatório mensal. "
            "Já reinstalei e o erro continua.",
            "team",
            _route(),
            ExpectedLabel(choice="technical"),
            "Fechamento reprodutível depois de reinstalar é defeito do produto.",
        ),
        _labeled(
            "route_login_email",
            "Preciso trocar o e-mail de login da conta da empresa. "
            "O endereço atual foi desativado.",
            "team",
            _route(),
            ExpectedLabel(choice="account"),
            "Mudança de dado cadastral, sem bug e sem disputa de cobrança.",
        ),
        _labeled(
            "tone_hostile",
            "Isso é um absurdo. Vou processar vocês se o dinheiro não voltar hoje.",
            "tone",
            _tone(),
            ExpectedLabel(choice="angry"),
            "Há ameaça e hostilidade. Não é elogio nem tom protocolar.",
        ),
        _labeled(
            "tone_polite",
            "Olá, poderiam confirmar o prazo de entrega do pedido 1842? Obrigado.",
            "tone",
            _tone(),
            ExpectedLabel(choice="calm"),
            "Pedido neutro, com saudação e agradecimento, sem carga emocional.",
        ),
        _labeled(
            "tone_excited",
            "Consegui! O relatório abriu e os números batem. Adorei a correção, ficou excelente.",
            "tone",
            _tone(),
            ExpectedLabel(choice="excited"),
            "Elogio entusiasmado. Não é hostilidade nem só uma fórmula de cortesia.",
        ),
        _labeled(
            "urgency_payroll_today",
            "O pagamento da folha falha hoje e os funcionários não recebem. "
            "Precisamos de correção ainda hoje.",
            "urgency",
            _urgency(),
            ExpectedLabel(score_level=2),
            "O texto marca o prazo como hoje e o impacto é imediato.",
        ),
        _labeled(
            "urgency_copy_when_possible",
            "Quando puderem, enviem a segunda via da nota do mês passado. Sem pressa.",
            "urgency",
            _urgency(),
            ExpectedLabel(score_level=0),
            "A pessoa dispensa pressa e não fixa data.",
        ),
        _labeled(
            "urgency_by_friday",
            "O relatório de conformidade precisa ser corrigido até sexta-feira desta semana.",
            "urgency",
            _urgency(),
            ExpectedLabel(score_level=1),
            "Há prazo dentro da semana. O texto não diz que precisa ser hoje.",
        ),
        _labeled(
            "frustration_exhausted",
            "Já liguei quatro vezes. Ninguém resolve. Estou exausto com esse suporte.",
            "frustration",
            _frustration(),
            ExpectedLabel(score_level=2),
            "Repetição sem solução e a palavra exausto estão no topo da rubrica.",
        ),
        _labeled(
            "refund_requested_yes",
            "Por favor, estornem a cobrança duplicada no cartão final 4242.",
            "refund_requested",
            _refund(),
            ExpectedLabel(noul=True),
            "O verbo pede estorno. O rótulo é sim; a probabilidade fica com o modelo.",
        ),
        _labeled(
            "refund_requested_no",
            "Como exporto os contatos para CSV? Não encontrei o botão.",
            "refund_requested",
            _refund(),
            ExpectedLabel(noul=False),
            "Pedido de instrução do produto, sem devolução de dinheiro.",
        ),
        _labeled(
            "duplicate_charge_yes",
            "A fatura lista a mensalidade duas vezes, nos dias 2 e 3, com o mesmo valor.",
            "duplicate_charge",
            _duplicate(),
            ExpectedLabel(noul=True),
            "Dois lançamentos do mesmo valor, descritos como a mesma mensalidade.",
        ),
        _labeled(
            "legitimate_boleto_not_spam",
            "Bom dia, o boleto venceu ontem e o código de barras não lê. Conseguem reenviar?",
            "unsolicited_ad",
            _spam(),
            ExpectedLabel(noul=False),
            "Cliente pede reenvio de boleto. Não há oferta nem captação.",
        ),
        DecisionTask(
            id="box_bumper_skipped",
            source="synthetic",
            state=(
                "Há uma foto do para-choque depois da batida, "
                "mas o arquivo não entra neste conjunto."
            ),
            questions={"damage": _box()},
            skip=True,
            skip_reason=(
                "Box depende de imagem. O PR1 não envia asset nem o header de imagens; "
                "a pergunta fica no conjunto só para o runner registrar o skip."
            ),
            note="Sem pixel público não há gabarito de coordenada.",
        ),
    ]


def _labeled(
    task_id: str,
    state: str,
    question_name: str,
    question: Question,
    label: ExpectedLabel,
    note: str,
) -> DecisionTask:
    return DecisionTask(
        id=task_id,
        source="synthetic",
        state=state,
        questions={question_name: question},
        expected={question_name: label},
        note=note,
    )


def _route() -> ChoiceQuestion:
    return ChoiceQuestion(
        instructions="Qual equipe deve cuidar deste chamado?",
        criteria={
            "billing": "Cobrança, fatura, estorno ou disputa de pagamento.",
            "technical": "Defeito, indisponibilidade ou bug do produto.",
            "account": "Login, perfil ou dado cadastral, sem defeito do produto.",
        },
    )


def _tone() -> ChoiceQuestion:
    return ChoiceQuestion(
        instructions="Qual é o tom desta mensagem?",
        criteria={
            "angry": "Mensagem hostil, irritada ou ameaçadora.",
            "calm": "Mensagem neutra ou educada, sem carga emocional forte.",
            "excited": "Mensagem entusiasmada ou de elogio animado.",
        },
    )


def _urgency() -> ScoreQuestion:
    return ScoreQuestion(
        instructions="Qual é a urgência deste pedido?",
        criteria=[
            "Pode esperar",
            "Precisa de atenção nesta semana",
            "Precisa de atenção hoje",
        ],
    )


def _frustration() -> ScoreQuestion:
    return ScoreQuestion(
        instructions="Qual é o nível de frustração de quem escreveu?",
        criteria=["Calmo", "Frustrado", "Muito frustrado"],
    )


def _refund() -> NoulQuestion:
    return NoulQuestion(
        instructions="Esta mensagem pede estorno ou reembolso?",
        criteria=NoulCriteria(
            true_case="A pessoa pede devolução de valor ou cancelamento de uma cobrança.",
            false_case="A pessoa não pede dinheiro de volta.",
        ),
    )


def _duplicate() -> NoulQuestion:
    return NoulQuestion(
        instructions="O relato indica cobrança duplicada?",
        criteria=NoulCriteria(
            true_case="O mesmo valor ou o mesmo item aparece cobrado mais de uma vez.",
            false_case="Não há dois lançamentos do mesmo item.",
        ),
    )


def _spam() -> NoulQuestion:
    return NoulQuestion(
        instructions="Esta mensagem é publicidade não solicitada?",
        criteria=NoulCriteria(
            true_case="A mensagem oferece algo ou capta contato sem ter sido pedida.",
            false_case="A mensagem é uma conversa legítima com a empresa.",
        ),
    )


def _box() -> BoxQuestion:
    return BoxQuestion(instructions="Onde está o dano no para-choque?")
